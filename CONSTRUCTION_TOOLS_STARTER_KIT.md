# MindFlow AS Construction Management Tools - Starter Kit

**Quick-start guide for building practical construction AI tools**

---

## Tool Overview

### 1. RF Takeoff Review Tool
**Purpose:** Automated material quantity extraction from construction plans

**Capabilities:**
- Identify and count materials (doors, windows, fixtures)
- Measure lengths, areas, and volumes
- Generate material lists with specifications
- Export to Excel/CSV for estimation

**Input:** PDF plans, images, or scanned drawings
**Output:** Structured material takeoff data

---

### 2. Plan Validation Tool
**Purpose:** Automated plan review for compliance and errors

**Capabilities:**
- Check building code compliance
- Identify design conflicts (electrical, plumbing, structural)
- Verify dimensions and specifications
- Flag missing or incorrect elements

**Input:** Construction plans (architectural, MEP, structural)
**Output:** Validation report with issues flagged

---

### 3. Engineering Change Impact Tool
**Purpose:** Analyze impact of design changes

**Capabilities:**
- Compare plan versions (before/after)
- Identify affected areas and systems
- Calculate scope of change
- Estimate cost impact

**Input:** Two versions of plans
**Output:** Change impact report with visualizations

---

### 4. Learning AI Plan Takeoff
**Purpose:** Continuously improving takeoff accuracy

**Capabilities:**
- Learn from user corrections
- Adapt to company-specific standards
- Improve recognition over time
- Build custom templates

**Input:** Plans + user feedback
**Output:** Improved takeoff accuracy over time

---

### 5. PO Review & Plan Review Tool
**Purpose:** Verify purchase orders match plan requirements

**Capabilities:**
- Cross-reference PO items with plan specs
- Verify quantities and specifications
- Flag discrepancies and substitutions
- Generate reconciliation reports

**Input:** Purchase orders + construction plans
**Output:** Verification report with flagged issues

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (Web Dashboard, CLI, Mobile App, API Endpoints)            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Application Logic Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ RF Takeoff   │  │   Plan       │  │   Change     │     │
│  │   Engine     │  │  Validator   │  │   Analyzer   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Learning    │  │ PO Matcher   │                        │
│  │   Engine     │  │   Engine     │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Vision AI Services Layer                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Unified Vision API Client                             │ │
│  │  - GPT-4o Vision  - Claude 3.5  - Gemini Vision      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Document Processing Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     PDF      │  │     DWG      │  │    Image     │     │
│  │   Parser     │  │   Parser     │  │  Processor   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Plan DB    │  │  Knowledge   │  │   Results    │     │
│  │  (Versions)  │  │    Base      │  │    Cache     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mindflow-construction-ai/
│
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── api_config.yaml           # API keys and endpoints
│   ├── prompt_templates.yaml     # Construction-specific prompts
│   └── validation_rules.yaml     # Building codes and standards
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── vision_client.py      # Unified vision API wrapper
│   │   ├── openai_adapter.py     # GPT-4o implementation
│   │   ├── anthropic_adapter.py  # Claude 3.5 implementation
│   │   └── gemini_adapter.py     # Gemini implementation
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rf_takeoff/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py      # Material extraction logic
│   │   │   ├── quantifier.py     # Quantity calculations
│   │   │   └── formatter.py      # Output formatting
│   │   │
│   │   ├── plan_validation/
│   │   │   ├── __init__.py
│   │   │   ├── validator.py      # Plan validation logic
│   │   │   ├── code_checker.py   # Building code compliance
│   │   │   └── conflict_detector.py  # Design conflict detection
│   │   │
│   │   ├── change_analysis/
│   │   │   ├── __init__.py
│   │   │   ├── comparator.py     # Plan version comparison
│   │   │   ├── impact_analyzer.py    # Change impact analysis
│   │   │   └── visualizer.py     # Change visualization
│   │   │
│   │   ├── learning_engine/
│   │   │   ├── __init__.py
│   │   │   ├── feedback_loop.py  # User feedback processing
│   │   │   ├── model_adapter.py  # Adaptive learning logic
│   │   │   └── template_builder.py   # Custom template creation
│   │   │
│   │   └── po_review/
│   │       ├── __init__.py
│   │       ├── matcher.py        # PO to plan matching
│   │       ├── verifier.py       # Specification verification
│   │       └── reporter.py       # Discrepancy reporting
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # PDF plan extraction
│   │   ├── dwg_parser.py         # AutoCAD file handling
│   │   ├── image_processor.py    # Image enhancement/preprocessing
│   │   └── ocr_engine.py         # Text extraction from scans
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── plan_repository.py    # Plan storage and versioning
│   │   ├── results_repository.py # Analysis results storage
│   │   └── knowledge_base.py     # Construction standards database
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── prompts.py            # Prompt engineering utilities
│   │   ├── cache.py              # Response caching
│   │   ├── retry.py              # Error handling and retry logic
│   │   └── logger.py             # Logging configuration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py         # File handling utilities
│       ├── image_utils.py        # Image processing helpers
│       ├── measurement_utils.py  # Unit conversion, calculations
│       └── export_utils.py       # Excel/CSV export functions
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── takeoff.py            # RF Takeoff endpoints
│   │   ├── validation.py         # Plan validation endpoints
│   │   ├── changes.py            # Change analysis endpoints
│   │   ├── po_review.py          # PO review endpoints
│   │   └── health.py             # Health check endpoints
│   │
│   └── models/
│       ├── __init__.py
│       ├── requests.py           # API request models
│       └── responses.py          # API response models
│
├── ui/
│   ├── web/
│   │   ├── app.py                # Streamlit dashboard
│   │   ├── pages/
│   │   │   ├── 1_RF_Takeoff.py
│   │   │   ├── 2_Plan_Validation.py
│   │   │   ├── 3_Change_Analysis.py
│   │   │   ├── 4_Learning_Engine.py
│   │   │   └── 5_PO_Review.py
│   │   └── components/
│   │       ├── plan_viewer.py
│   │       ├── results_table.py
│   │       └── charts.py
│   │
│   └── cli/
│       ├── __init__.py
│       └── menu.py               # Command-line interface
│
├── tests/
│   ├── __init__.py
│   ├── test_api/
│   ├── test_tools/
│   ├── test_parsers/
│   └── test_integration/
│
├── scripts/
│   ├── setup_database.py         # Database initialization
│   ├── import_standards.py       # Import building codes
│   └── batch_process.py          # Batch plan processing
│
├── docs/
│   ├── API.md                    # API documentation
│   ├── USER_GUIDE.md             # User guide
│   ├── DEPLOYMENT.md             # Deployment instructions
│   └── PROMPTS.md                # Prompt engineering guide
│
└── data/
    ├── standards/                # Building codes and standards
    ├── templates/                # Report templates
    └── samples/                  # Sample plans for testing
```

---

## Quick Start Code Examples

### Example 1: RF Takeoff Tool

```python
# src/tools/rf_takeoff/extractor.py

from src.api.vision_client import VisionClient
from src.core.prompts import TAKEOFF_PROMPT_TEMPLATE
from typing import Dict, List
import json

class MaterialExtractor:
    def __init__(self):
        self.vision_client = VisionClient()

    def extract_materials(self, plan_image: str, material_type: str = "all") -> Dict:
        """Extract material quantities from construction plan."""

        prompt = TAKEOFF_PROMPT_TEMPLATE.format(
            material_type=material_type,
            instructions="""
            Analyze this construction plan and extract:
            1. Material types (doors, windows, fixtures, etc.)
            2. Quantities for each material
            3. Specifications (sizes, materials, types)
            4. Locations (room names or grid coordinates)

            Return results in JSON format:
            {
                "materials": [
                    {
                        "type": "door",
                        "subtype": "interior",
                        "quantity": 5,
                        "specification": "36\" x 80\" hollow core",
                        "locations": ["Room 101", "Room 102"]
                    }
                ]
            }
            """
        )

        response = self.vision_client.analyze_image(
            image_path=plan_image,
            prompt=prompt,
            response_format="json"
        )

        return self._parse_and_validate(response)

    def _parse_and_validate(self, response: str) -> Dict:
        """Parse and validate the extraction results."""
        try:
            data = json.loads(response)
            # Add validation logic here
            return data
        except json.JSONDecodeError:
            # Handle parsing errors
            return {"error": "Failed to parse response"}

# Usage
extractor = MaterialExtractor()
results = extractor.extract_materials("floor_plan.pdf", material_type="doors")
print(f"Found {len(results['materials'])} door types")
```

### Example 2: Plan Validation Tool

```python
# src/tools/plan_validation/validator.py

from src.api.vision_client import VisionClient
from src.database.knowledge_base import BuildingCodeDB
from typing import List, Dict

class PlanValidator:
    def __init__(self):
        self.vision_client = VisionClient()
        self.code_db = BuildingCodeDB()

    def validate_plan(self, plan_image: str, plan_type: str, jurisdiction: str) -> Dict:
        """Validate construction plan against building codes."""

        # Get applicable codes for jurisdiction
        applicable_codes = self.code_db.get_codes(jurisdiction, plan_type)

        prompt = f"""
        Analyze this {plan_type} construction plan for compliance with building codes.

        Check for:
        1. Egress requirements (doors, exits, corridors)
        2. Room dimensions and clearances
        3. Accessibility requirements (ADA compliance)
        4. Fire safety (sprinklers, fire-rated walls)
        5. Structural elements (beams, columns, load paths)

        Applicable codes: {applicable_codes}

        Return findings in JSON format:
        {{
            "compliant": true/false,
            "issues": [
                {{
                    "category": "egress",
                    "severity": "high/medium/low",
                    "description": "Corridor width below minimum 44 inches",
                    "location": "Grid B-3 to B-5",
                    "code_reference": "IBC 2021 Section 1020.2",
                    "recommendation": "Increase corridor width to 44 inches minimum"
                }}
            ],
            "summary": "Overall compliance status"
        }}
        """

        response = self.vision_client.analyze_image(
            image_path=plan_image,
            prompt=prompt,
            response_format="json"
        )

        return self._generate_report(response)

    def _generate_report(self, validation_results: str) -> Dict:
        """Generate formatted validation report."""
        # Report generation logic
        pass

# Usage
validator = PlanValidator()
report = validator.validate_plan(
    plan_image="architectural_plan.pdf",
    plan_type="architectural",
    jurisdiction="IBC_2021"
)
```

### Example 3: Change Impact Analyzer

```python
# src/tools/change_analysis/comparator.py

from src.api.vision_client import VisionClient
from typing import Dict, Tuple
import difflib

class ChangeAnalyzer:
    def __init__(self):
        self.vision_client = VisionClient()

    def analyze_changes(self,
                       original_plan: str,
                       revised_plan: str,
                       change_description: str = "") -> Dict:
        """Compare two plan versions and analyze impact."""

        # First, get descriptions of both plans
        original_desc = self._describe_plan(original_plan)
        revised_desc = self._describe_plan(revised_plan)

        # Then compare them
        prompt = f"""
        Compare these two construction plan versions:

        ORIGINAL PLAN DESCRIPTION:
        {original_desc}

        REVISED PLAN DESCRIPTION:
        {revised_desc}

        CHANGE ORDER DESCRIPTION:
        {change_description}

        Identify:
        1. What changed (additions, deletions, modifications)
        2. Affected systems (structural, electrical, plumbing, HVAC)
        3. Impact scope (major, moderate, minor)
        4. Downstream impacts (schedule, cost, other trades)
        5. Required actions (re-engineering, permit updates, material changes)

        Return in JSON format with detailed change analysis.
        """

        response = self.vision_client.analyze_multi_image(
            images=[original_plan, revised_plan],
            prompt=prompt,
            response_format="json"
        )

        return self._calculate_impact_score(response)

    def _describe_plan(self, plan_image: str) -> str:
        """Generate detailed description of plan."""
        prompt = """
        Describe this construction plan in detail:
        - Layout and dimensions
        - Key elements and their locations
        - Systems shown (electrical, plumbing, etc.)
        - Materials and specifications
        - Notes and callouts
        """
        return self.vision_client.analyze_image(plan_image, prompt)

    def _calculate_impact_score(self, change_data: Dict) -> Dict:
        """Calculate quantitative impact score."""
        # Impact scoring logic
        pass

# Usage
analyzer = ChangeAnalyzer()
impact = analyzer.analyze_changes(
    original_plan="plan_v1.pdf",
    revised_plan="plan_v2.pdf",
    change_description="Relocate main entrance to north wall"
)
```

---

## Configuration Files

### config/api_config.yaml

```yaml
# Vision API Configuration
vision_apis:
  primary: "openai"  # or "anthropic" or "gemini"
  fallback: "anthropic"

  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o"
    max_tokens: 4096
    temperature: 0.1

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"
    max_tokens: 4096
    temperature: 0.1

  gemini:
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-pro-vision"
    max_tokens: 4096
    temperature: 0.1

# Rate limiting
rate_limits:
  requests_per_minute: 60
  tokens_per_minute: 150000
  retry_attempts: 3
  retry_delay: 2

# Caching
cache:
  enabled: true
  ttl_seconds: 3600
  max_size_mb: 1000
```

### config/prompt_templates.yaml

```yaml
# RF Takeoff Prompts
rf_takeoff:
  doors: |
    Analyze this floor plan and extract all door information:
    - Count total doors
    - Identify door types (interior, exterior, fire-rated)
    - Measure door dimensions (width x height)
    - Note door materials and specifications
    - Record room locations
    Format: JSON with structured data

  windows: |
    Extract window information from this plan:
    - Count windows by type and size
    - Identify window specifications
    - Note locations and elevations
    Format: JSON

  materials: |
    Extract material quantities for: {material_type}
    Include: type, quantity, specifications, locations
    Format: JSON

# Plan Validation Prompts
plan_validation:
  egress: |
    Check egress requirements for {building_type}:
    - Exit door widths (minimum 36")
    - Corridor widths (minimum 44")
    - Travel distances to exits
    - Exit signage locations
    Code: {building_code}

  accessibility: |
    Verify ADA compliance:
    - Accessible routes
    - Door clearances
    - Restroom accessibility
    - Parking spaces
    Report violations with code references

# Change Analysis Prompts
change_analysis:
  comparison: |
    Compare these plan versions systematically:
    1. Identify all visible changes
    2. Categorize by system type
    3. Assess impact severity
    4. Flag regulatory impacts
    5. Estimate scope of rework
```

---

## requirements.txt

```txt
# Core dependencies
python>=3.10

# Vision APIs
openai>=1.12.0
anthropic>=0.18.0
google-generativeai>=0.3.0

# Web framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
streamlit>=1.31.0

# Document processing
PyMuPDF>=1.23.0  # PDF parsing
pdfplumber>=0.10.0
ezdxf>=1.1.0  # DWG/DXF files
Pillow>=10.2.0  # Image processing
opencv-python>=4.9.0
pytesseract>=0.3.10  # OCR

# Database
sqlalchemy>=2.0.25
alembic>=1.13.0
psycopg2-binary>=2.9.9  # PostgreSQL

# Data processing
pandas>=2.2.0
numpy>=1.26.0
openpyxl>=3.1.2  # Excel export

# Utilities
pydantic>=2.6.0
python-dotenv>=1.0.0
pyyaml>=6.0.1
httpx>=0.26.0
tenacity>=8.2.3  # Retry logic
redis>=5.0.1  # Caching

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-asyncio>=0.23.0

# Development
black>=24.1.0
ruff>=0.2.0
mypy>=1.8.0
```

---

## Environment Setup

### .env.example

```bash
# Vision API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/construction_ai
REDIS_URL=redis://localhost:6379/0

# Application
APP_ENV=development
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Feature Flags
ENABLE_CACHING=true
ENABLE_LEARNING_ENGINE=true
ENABLE_BATCH_PROCESSING=true

# Construction Standards
DEFAULT_BUILDING_CODE=IBC_2021
DEFAULT_JURISDICTION=US
DEFAULT_UNITS=imperial
```

---

## Deployment Options

### Option 1: Local Development

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run API
uvicorn api.main:app --reload --port 8000

# Run Web UI (separate terminal)
streamlit run ui/web/app.py
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/construction_ai
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    depends_on:
      - db
      - redis

  web:
    build: .
    command: streamlit run ui/web/app.py
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=construction_ai
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Option 3: Cloud Deployment (AWS)

```bash
# Using AWS ECS/Fargate
# 1. Build and push Docker image to ECR
# 2. Create ECS task definition
# 3. Deploy to ECS cluster
# 4. Configure Application Load Balancer
# 5. Set up RDS PostgreSQL and ElastiCache Redis

# Terraform configuration available in docs/DEPLOYMENT.md
```

---

## Next Steps

### Week 1: Proof of Concept
1. Set up API accounts (OpenAI/Anthropic)
2. Test with 5-10 sample construction plans
3. Evaluate accuracy and API costs
4. Document findings

### Week 2: MVP Development
1. Implement RF Takeoff tool (priority 1)
2. Build simple API endpoints
3. Create basic web interface
4. Add error handling

### Week 3-4: Expand Features
1. Add Plan Validation tool
2. Implement Change Analysis
3. Build PO Review tool
4. Add export functionality

### Week 5-6: Integration
1. Connect to existing systems
2. Add batch processing
3. Implement caching
4. Optimize performance

### Week 7-8: Beta Testing
1. Deploy to staging environment
2. Test with real users
3. Gather feedback
4. Refine and improve

---

## Cost Estimation

### API Costs (GPT-4o Vision)
- **Input:** $0.0025 per 1K tokens (~$0.003 per image)
- **Output:** $0.01 per 1K tokens
- **Monthly estimate (1000 plans):** $100-$300

### Infrastructure Costs
- **Hosting:** $50-$100/month (Digital Ocean, AWS Lightsail)
- **Database:** $20-$50/month (managed PostgreSQL)
- **Caching:** $10-$20/month (Redis)
- **Total:** $180-$470/month

### Total First Year
- **Development:** $40K-$60K (2-3 months, 1-2 devs)
- **Operations:** $2K-$6K (hosting + APIs)
- **Maintenance:** $10K-$20K (updates, support)
- **TOTAL:** $52K-$86K

---

## Support & Resources

### Community
- Construction AI Slack/Discord
- Weekly office hours for questions
- GitHub discussions for feature requests

### Documentation
- API documentation: `/docs/API.md`
- User guide: `/docs/USER_GUIDE.md`
- Deployment guide: `/docs/DEPLOYMENT.md`
- Prompt engineering tips: `/docs/PROMPTS.md`

### Contact
- Technical questions: tech@mindflow.as
- Feature requests: product@mindflow.as
- Bug reports: GitHub Issues

---

**Ready to get started? Begin with the Proof of Concept phase!**
