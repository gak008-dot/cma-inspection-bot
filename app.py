import chainlit as cl
from chainlit.input_widget import Select
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_FILE   = "CMA INSPECTION - Extended - Survey Guidance notes for Surveyors - v2025.1.pdf"
PAGES_DIR  = "pdf_pages"
Q_MAP_FILE = "question_page_map.json"
MODEL      = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ── Constants ──────────────────────────────────────────────────────────────────
SECTIONS = {
    "1":  ("Documentation",  "1.1 – 1.49"),
    "2":  ("Chief Officer",  "2.1 – 2.15"),
    "3":  ("Chief Engineer", "3.1 – 3.45"),
    "4":  ("General",        "4.1 – 4.8"),
    "5":  ("LSA / FFE",      "5.1 – 5.13"),
    "6":  ("Accommodation",  "6.1 – 6.10"),
    "7":  ("Bridge",         "7.1 – 7.12"),
    "8":  ("Hull & Deck",    "8.1 – 8.27"),
    "9":  ("Cargo Gear",     "9.1 – 9.11"),
    "10": ("Engine Room",    "10.1 – 10.47"),
}

CORROSION_STAGES = """## Corrosion Assessment Guide

**Stage 1 — Superficial** ✅ Answer: YES
Light brown, atmospheric rust. Brushes off easily.
No structural effect. Crew can maintain underway.

**Stage 2 — Progressive** ✅ Answer: YES (note in summary)
Darker rust. Some loose scales beginning to detach.
Action needed but not critical. Normal maintenance regime.

**Stage 3 — Excessive** ❌ Answer: NO
Dark brown. Loose scales on/around structure.
Comment: *"excessive corrosion"*
Shore assistance may be needed.

**Stage 4 — Wastage** ❌ Answer: NO
Material lost. Structure holed or laminar corrosion.
Comment: *"wastage"*
Shore assistance required.
"""

KEY_VALUES = """## Key Technical Values

**Main Engine**
- Sump LO minimum: `1.3 L/kW` (pumps running)
- Sump LO minimum: `90%` capacity (pumps stopped)
- LO water content limit: `< 0.3%`
- Monthly performance checks required

**Auxiliary Engines**
- Load test minimum: `70% MCR` monthly
- \>14,000 TEU: AE#1/#4 = `50%`, AE#2/#3 = `60%`
- Test period review: last `4 months`

**OWS / Bilge**
- Overboard discharge limit: `15 ppm`
- Earth fault minimum: `1 MΩ`

**LSA / Safety**
- Lifeboat launch interval: `3 months`
- SOPEP drill interval: `3 months`
- Fire/abandon ship drill: `monthly`
- Safety meetings: `4–6 weeks`

**Emergency Generator**
- Fuel minimum: `18 hours` uninterrupted

**Lashing**
- Lashing force limit: `100%` (loading computer)
- Inventory check interval: `6 months`
- PMS overdue tolerance: `< 5%`

**Crew Rest (MLC 2006)**
- Daily minimum: `10 hours`
- Weekly minimum: `77 hours`
- Max consecutive work: `14 hours`
"""

SECTION_QUICK_TIPS = {
    "1":  "Check PSC report, ISM audits, certificates list, drug/alcohol policy, drills records.",
    "2":  "Check deck log, lashing forces <100%, lashing inventory within 6 months, bilge alarms monthly.",
    "3":  "Check PMS <5% overdue, ME performance monthly, AE at 70% MCR, OWS records, bunker procedures.",
    "4":  "Fire flaps, save-alls oil-free, ladders safe, SOPEP kit stocked.",
    "5":  "LSA/FFE validity, lifeboat engine start test, emergency fire pump 20m reach, PPE compliance.",
    "6":  "Galley clean, cold store lock-in alarm, medicines in date, garbage segregated.",
    "7":  "ECDIS up to date, BNWAS active, passage planning berth-to-berth with UKC.",
    "8":  "Corrosion stages, hatch covers, lashing gear type consistency, mooring lines condition.",
    "9":  "Crane pedestals, wires >10% broken strands = discard, SWL marked on jib.",
    "10": "ER housekeeping, bilge oil-free, OWS 3-way valve test, switchboard earth faults <1MΩ.",
}

REFERENCE_PAGES = {
    "corrosion": [7, 8],
    "general":   [2, 3, 4, 5],
}

# ── Setup functions ────────────────────────────────────────────────────────────
def ensure_pages_extracted():
    os.makedirs(PAGES_DIR, exist_ok=True)
    map_file = Path(PAGES_DIR) / "page_map.json"

    if map_file.exists():
        with open(map_file) as f:
            return json.load(f)

    print("Extracting PDF pages as images...")
    doc = fitz.open(PDF_FILE)
    page_map = {}

    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)
        img_path = str(Path(PAGES_DIR) / f"page_{i+1:03d}.png")
        pix.save(img_path)
        page_map[str(i + 1)] = img_path

    doc.close()
    with open(map_file, "w") as f:
        json.dump(page_map, f)

    print(f"Extracted {len(page_map)} pages")
    return page_map


def ensure_question_map():
    if Path(Q_MAP_FILE).exists():
        with open(Q_MAP_FILE) as f:
            return json.load(f)

    print("Building question→page map...")
    doc = fitz.open(PDF_FILE)
    pattern = re.compile(r'\b(\d{1,2}\.\d{1,2})\b')
    question_map = {}

    for i in range(len(doc)):
        text = doc[i].get_text()
        for match in pattern.findall(text):
            if match not in question_map:
                question_map[match] = []
            if (i + 1) not in question_map[match]:
                question_map[match].append(i + 1)

    doc.close()
    with open(Q_MAP_FILE, "w") as f:
        json.dump(question_map, f)

    print(f"Mapped {len(question_map)} questions")
    return question_map


@cl.cache
def load_vector_db():
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_question_numbers(text: str) -> list:
    cleaned = re.sub(r'(?i)(question|q\.?)\s*', '', text)
    pattern = re.compile(r'\b(\d{1,2}\.\d{1,2})\b')
    return list(set(pattern.findall(cleaned)))


def is_corrosion_related(text: str) -> bool:
    keywords = ["corrosion", "rust", "wastage", "corroded", "scale", "paint", "stage"]
    return any(k in text.lower() for k in keywords)


def get_pages_for_questions(question_nums, question_map, page_map,
                             include_corrosion=False):
    pages = set()
    for q in question_nums:
        for page in question_map.get(q, []):
            pages.add(page)
    if include_corrosion:
        for p in REFERENCE_PAGES["corrosion"]:
            pages.add(p)

    result = []
    for page_num in sorted(pages):
        img_path = page_map.get(str(page_num))
        if img_path and Path(img_path).exists():
            result.append((page_num, img_path))
    return result


def get_section_from_questions(question_nums: list) -> str:
    if not question_nums:
        return ""
    return question_nums[0].split(".")[0]


def build_sidebar_content(active_section: str = "") -> str:
    lines = ["### 📋 Inspection Sections\n"]
    for num, (name, qrange) in SECTIONS.items():
        marker = "→ " if num == active_section else "   "
        lines.append(f"{marker}**{num}. {name}**")
        lines.append(f"   *{qrange}*")
        if num == active_section and num in SECTION_QUICK_TIPS:
            lines.append(f"\n   💡 {SECTION_QUICK_TIPS[num]}\n")
    return "\n".join(lines)


# ── Chainlit lifecycle ─────────────────────────────────────────────────────────
@cl.on_chat_start
async def start():
    # Setup
    await cl.Message(content="⏳ Loading inspection database and extracting PDF pages...").send()

    page_map     = ensure_pages_extracted()
    question_map = ensure_question_map()
    db           = load_vector_db()

    cl.user_session.set("db",            db)
    cl.user_session.set("page_map",      page_map)
    cl.user_session.set("question_map",  question_map)
    cl.user_session.set("history",       [])
    cl.user_session.set("active_section", "")

    # Section selector widget
    await cl.ChatSettings(
        inputs=[
            Select(
                id="active_section",
                label="📂 Jump to Section",
                values=[""] + list(SECTIONS.keys()),
                initial_value="",
                description="Select section to prime the assistant context"
            )
        ]
    ).send()

    # Sidebar — corrosion guide
    await cl.Message(
        content=CORROSION_STAGES,
        author="📊 Corrosion Guide"
    ).send()

    # Sidebar — key values
    await cl.Message(
        content=KEY_VALUES,
        author="🔢 Key Values"
    ).send()

    # Welcome
    await cl.Message(
        content="""## ⚓ CMA CGM Inspection Assistant
**v2025.1 — 58 pages loaded with photos**

---
**How to use:**
- Type a question number: `8.2` or `Q 8.2` or `question 8.2`
- Type a topic: `OWS three-way valve test` or `lashing forces`
- Type a section: `start section 5` or `LSA checks`
- Ask corrosion: `stage 3 corrosion on hatch coaming`

**I will show:**
✅ Structured Y/N/N/A guidance  
📄 Relevant PDF pages with photos  
📋 Exact comment text to record  
""",
        author="⚓ Assistant"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    section = settings.get("active_section", "")
    cl.user_session.set("active_section", section)

    if section and section in SECTIONS:
        name, qrange = SECTIONS[section]
        tip = SECTION_QUICK_TIPS.get(section, "")
        await cl.Message(
            content=f"📂 **Section {section}: {name}** ({qrange})\n\n💡 {tip}\n\nAsk me about any question in this section.",
            author="⚓ Assistant"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    db            = cl.user_session.get("db")
    page_map      = cl.user_session.get("page_map")
    question_map  = cl.user_session.get("question_map")
    history       = cl.user_session.get("history")
    active_section = cl.user_session.get("active_section", "")

    user_query = message.content

    # Detect section change in message
    section_match = re.search(r'section\s*(\d{1,2})', user_query, re.IGNORECASE)
    if section_match:
        active_section = section_match.group(1)
        cl.user_session.set("active_section", active_section)

    # Extract question numbers
    question_nums   = extract_question_numbers(user_query)
    corrosion_query = is_corrosion_related(user_query)

    # If no specific question but section is active, note it
    detected_section = get_section_from_questions(question_nums) or active_section
    section_info = ""
    if detected_section in SECTIONS:
        name, qrange = SECTIONS[detected_section]
        section_info = f"Active section: {detected_section}. {name} ({qrange})."

    # Retrieve text
    retrieved_docs = db.similarity_search(user_query, k=6)
    context = "\n\n".join([
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in retrieved_docs
    ])

    # System prompt
    system_prompt = f"""You are a senior BMT surveyor and maritime technical superintendent 
conducting a CMA CGM Extended Vessel Condition Inspection on behalf of CMA Ships.

{section_info}

ALWAYS structure your response exactly like this:

---
**Question [X.X] — [Full question title from guidance]**

**Answer category:** Y / N / N/A / N/I

**What to physically check:**
[Bullet points of specific things to look for]

**How to decide:**
[Clear decision logic from the guidance]

**If answering N — record this comment (≤150 chars):**
[Example: "Various lashing bridges had excessive corrosion, locally wastage" NOT "corroded"]

**Section summary note (if needed):**
[Anything worth noting in the final summary even if answer is Y]
---

CORROSION — always use the 4-stage scale:
- Stage 1 Superficial (light brown, brushes off) → Y
- Stage 2 Progressive (darker, some loose scales) → Y, note in summary
- Stage 3 Excessive (dark brown, large area, loose scales) → N, comment: excessive corrosion
- Stage 4 Wastage (holes, laminar, material lost) → N, comment: wastage

TEST ITEMS — give numbered steps exactly as per guidance.

KEY VALUES — always state exact numbers:
- ME sump LO: 1.3 L/kW (pumps on) | 90% capacity (pumps off)
- AE load test: 70% MCR monthly | >14K TEU: AE#1/#4=50%, AE#2/#3=60%
- LO water content: <0.3%
- OWS limit: 15 ppm
- Earth fault: minimum 1 MΩ
- Emergency generator fuel: 18 hours
- Lifeboat launch: every 3 months
- SOPEP drill: every 3 months
- Safety meetings: every 4–6 weeks
- Crew rest: 10h/day, 77h/week (MLC 2006)
- PMS overdue tolerance: <5%
- Lashing inventory: check every 6 months

Answer ONLY from the guidance context. Cite question numbers explicitly.
If the answer requires inspecting multiple sub-items, list each.

--- GUIDANCE CONTEXT ---
{context}
"""

    # Build message list with history
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_query})

    # Stream response
    response_msg = cl.Message(content="", author="⚓ Assistant")
    await response_msg.send()

    full_response = ""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        max_tokens=1500,
        temperature=0.1,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_response += token
        await response_msg.stream_token(token)

    await response_msg.update()

    # Show PDF pages
    pages_to_show = get_pages_for_questions(
        question_nums, question_map, page_map,
        include_corrosion=corrosion_query
    )

    # Fallback: corrosion query but no specific question
    if not pages_to_show and corrosion_query:
        for p in REFERENCE_PAGES["corrosion"]:
            img = page_map.get(str(p))
            if img and Path(img).exists():
                pages_to_show.append((p, img))

    if pages_to_show:
        await cl.Message(
            content="📄 **Reference pages from guidance manual:**",
            author="📄 PDF Reference"
        ).send()

        for page_num, img_path in pages_to_show[:4]:
            elements = [
                cl.Image(
                    path=img_path,
                    name=f"Page {page_num}",
                    display="inline"
                )
            ]
            await cl.Message(
                content=f"*Page {page_num} of the guidance*",
                elements=elements,
                author="📄 PDF Reference"
            ).send()

    # Update history
    history.append({"user": user_query, "assistant": full_response})
    if len(history) > 6:
        history = history[-6:]
    cl.user_session.set("history", history)
