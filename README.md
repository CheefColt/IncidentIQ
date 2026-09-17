# IncidentIQ

IncidentIQ is an evidence-grounded incident investigation system for log data.
It combines lexical retrieval, semantic retrieval, deterministic evidence analysis,
and structured Gemini reasoning to help investigate operational incidents.

Given a query such as `machine check timeout`, IncidentIQ retrieves relevant log
events, organizes them into an incident context, extracts observable patterns, and
asks an LLM to produce a structured analysis. When the supplied evidence is not
enough, the model can call a `search_logs` tool to retrieve additional evidence.
The resulting observations, hypotheses, and investigation steps are checked against
the evidence that was actually retrieved.

> **Status:** Experimental and under active development. The current repository is
> best suited for learning, evaluation, and prototyping rather than production use.

## What It Does

The current investigation pipeline is:

```text
User query
		|
		v
Hybrid retrieval (BM25 + semantic retrieval + RRF)
		|
		v
Structured context (evidence, timeline, groups, statistics)
		|
		v
Deterministic pattern extraction
		|
		v
Gemini structured analysis
		|
		+--> search_logs() when more evidence is needed
		|
		v
Evidence citation validation
		|
		v
InvestigationResult
```

The project deliberately keeps deterministic computation separate from LLM
reasoning. Retrieval and pattern extraction produce inspectable intermediate
artifacts, while Gemini is responsible for interpreting the evidence and deciding
whether further search is useful.

## Current Capabilities

- BM25 lexical retrieval for terminology and exact log vocabulary.
- Semantic retrieval using `all-MiniLM-L6-v2` for similar wording.
- Reciprocal Rank Fusion (RRF) to combine lexical and semantic rankings.
- Context construction with evidence records, message groups, timelines,
	statistics, time ranges, and temporal clusters.
- Deterministic extraction of repeated messages, affected-node counts, severity
	distributions, and temporal patterns.
- Structured incident analysis with:
	- summary
	- observations
	- hypotheses with confidence values
	- unknowns
	- concrete next investigation steps
- Gemini function calling through `search_logs` for iterative evidence gathering.
- Evidence provenance that keeps initial retrieval and tool-retrieved evidence
	separately.
- Citation validation that reports whether referenced document IDs were actually
	retrieved.
- A FastAPI endpoint at `POST /investigate`.

## Data

The repository currently uses a processed subset of the BGL supercomputer log
dataset:

```text
data/raw/BGL_2k.log
data/processed/logs.parquet
```

The processed corpus contains 2,000 log events with fields such as:

```text
log_id, label, timestamp, node, type, component, severity, message
```

The search and context layers use the processed Parquet file. If you replace the
corpus, preserve the fields required by the retrieval and context code, especially
`log_id`, `timestamp`, `node`, `severity`, and `message`.

## Requirements

- Python 3.10 or newer
- A Gemini API key
- Dependencies used by the current implementation:
	- `fastapi`
	- `google-genai`
	- `numpy`
	- `pandas`
	- `pyarrow`
	- `python-dotenv`
	- `pydantic`
	- `sentence-transformers`
	- an ASGI server such as `uvicorn`

The dependency list is currently maintained manually while the project is being
developed; `pyproject.toml` does not yet declare runtime dependencies.

## Setup

From the repository root, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the current runtime dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install fastapi google-genai numpy pandas pyarrow python-dotenv pydantic sentence-transformers uvicorn
```

Create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or expose the API key in source control.

## Using the Python Pipeline

The main Python interface is `IncidentIQ`:

```python
from incidentiq.pipeline import IncidentIQ

incidentiq = IncidentIQ(
		data_path="data/processed/logs.parquet",
)

result = incidentiq.investigate(
		"machine check timeout",
		top_k=10,
)

print(result.analysis.summary)
print(result.analysis.hypotheses)
print(result.grounding.is_grounded)
```

`investigate()` returns an `InvestigationResult` containing the original query,
the structured analysis, initial evidence, tool evidence, extracted patterns, and
a grounding report.

## Running the API

Start the FastAPI application from the repository root:

```powershell
uvicorn incidentiq.api.main:app --reload --app-dir src
```

The interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

Send an investigation request:

```powershell
Invoke-RestMethod `
	-Uri http://127.0.0.1:8000/investigate `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"query":"machine check timeout"}'
```

The API currently loads `data/processed/logs.parquet` from the repository path
configured in `src/incidentiq/api/routes.py`.

## Project Structure

```text
src/incidentiq/
|-- api/          FastAPI application and routes
|-- context/      Evidence context, signals, and pattern extraction
|-- evaluation/   Grounding validation and evaluation metrics
|-- indexing/     Corpus indexing and tokenization
|-- ingestion/    Log inspection utilities
|-- ranking/      Ranking combination, including RRF
|-- reasoning/    Gemini integration, prompts, and response models
|-- retrieval/    BM25, semantic, positional, and structured retrieval
|-- tools/        Search tools exposed to the reasoning model
|-- pipeline.py   End-to-end IncidentIQ investigation orchestration
|-- search.py     Public hybrid search interface
```

The notebooks contain the exploratory work behind lexical retrieval, semantic
retrieval, hybrid retrieval, evaluation, context construction, and reasoning.

## Evaluation and Reliability

The project includes building blocks for measuring retrieval and reasoning quality,
but it does not yet have a complete investigation benchmark. The current grounding
check verifies that evidence IDs cited in observations, hypotheses, and next steps
belong to the initial or tool-retrieved evidence set.

Grounding validation is evidence provenance validation, not proof that a conclusion
is factually correct. The system is still experimental, and model output, search
quality, latency, retries, and failure handling need broader evaluation.

## Roadmap

### Phase 1: Reliability

- Build a repeatable investigation evaluation harness.
- Represent multiple tool calls and deduplicate tool evidence.
- Define behavior when grounding validation fails.
- Measure retrieval quality, grounding, tool usage, and investigation quality.
- Improve timeout, retry, and error handling.

### Phase 2: Investigation quality

- Turn unknowns into focused, executable investigation actions.
- Refine hypotheses as new evidence is gathered.
- Compare evidence across successive searches.
- Make next steps useful both to an investigator and to the agent.

### Phase 3: Production engineering

- Add configuration instead of hard-coded paths and settings.
- Add request IDs, structured logging, latency metrics, token usage, and tool
	telemetry.
- Improve caching, concurrency, model configuration, and resource management.
- Declare and pin project dependencies.

### Phase 4: Advanced capabilities

Introduce reranking, approximate nearest-neighbor indexes, graph-based context,
or an orchestration framework only when measurements show that the current design
has reached a meaningful limitation.

## Design Principles

- Evidence before explanation.
- Deterministic calculations before probabilistic reasoning.
- Explicit provenance for every retrieved event.
- Hypotheses must be distinguished from observations.
- Unknowns are valuable outputs, not failures to hide.
- New infrastructure should be justified by a measured problem.

## License

No license has been declared yet.
