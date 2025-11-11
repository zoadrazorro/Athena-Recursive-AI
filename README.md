# Athena Recursive AI

**An agentic Meta-LLM architecture implementing Mixture of Experts (MoE) with Global Workspace Theory-inspired coordination for dual AMD 7900 XT GPUs.**

## Overview

Athena implements a hierarchical meta-reasoning system where a central orchestrator model coordinates specialized expert models, each fine-tuned or prompted for specific cognitive domains. This architecture is inspired by Global Workspace Theory (GWT) from cognitive science, implementing a computational model of consciousness where information competes for attention and is broadcast to specialized cognitive processes.

### Key Features

- **Multi-Expert Architecture**: Four specialized expert models (Reasoning, Creative, Technical, Memory)
- **Intelligent Query Routing**: Automatic analysis and routing to appropriate experts
- **Global Workspace**: GWT-inspired attention mechanism for information sharing
- **Parallel Consultation**: Multiple experts can process queries simultaneously
- **Conflict Resolution**: Weighted confidence scoring and consensus building
- **Dual GPU Support**: Optimized for running on dual AMD 7900 XT configuration (40GB total VRAM)

## Architecture

### Layer 1: Meta-Orchestrator (Primary Decision Layer)

The orchestrator (Qwen2.5 14B Q5_K_M, ~10GB VRAM on GPU 1) serves as the central reasoning hub:
- Analyzes incoming queries
- Decomposes complex problems
- Routes tasks to appropriate experts
- Synthesizes responses from multiple experts
- Maintains conversation state

### Layer 2: Expert Specialist Models (Domain-Specific Processing)

Four specialized models run on GPU 2 (4-bit quantization, ~18GB total):

1. **Reasoning Specialist** (Phi-3.5-mini Q4_K_M, ~3GB)
   - Logical inference and deductive reasoning
   - Mathematical problem-solving
   - Structured analytical thinking

2. **Creative Synthesis Specialist** (Mistral 7B Q4_K_M, ~4.5GB)
   - Creative ideation and brainstorming
   - Philosophical exploration
   - Conceptual blending and metaphor

3. **Technical Implementation Specialist** (CodeQwen 7B Q4_K_M, ~4.5GB)
   - Code generation and debugging
   - Software architecture design
   - Technical documentation

4. **Memory and Context Specialist** (Llama 3.1 8B Q4_K_M, ~5GB)
   - Conversation continuity
   - Context retrieval
   - Consistency checking

### Global Workspace

Implements attention-based information sharing:
- Experts compete for limited workspace slots based on attention weights
- High-confidence information is broadcast system-wide
- Temporal decay ensures fresh information takes priority
- Expert activations track current system focus

## Installation

### Prerequisites

- Python 3.10 or higher
- Dual AMD 7900 XT GPUs (or similar configuration with 40GB total VRAM)
- LM Studio installed and configured
- ROCm drivers for AMD GPUs (if on Linux)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/zoadrazorro/Athena-Recursive-AI.git
cd Athena-Recursive-AI
```

2. **Install dependencies:**

```bash
pip install -e .
# Or for development:
pip install -e ".[dev]"
```

3. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your LM Studio endpoints and settings
```

4. **Set up LM Studio:**

Load the following models in LM Studio:

- **GPU 1 (Port 1234)**: Qwen2.5 14B Instruct (Q5_K_M)
- **GPU 2 (Port 1235)**: Phi-3.5-mini Instruct (Q4_K_M)
- **GPU 2 (Port 1236)**: Mistral 7B Instruct (Q4_K_M)
- **GPU 2 (Port 1237)**: CodeQwen 7B Instruct (Q4_K_M)
- **GPU 2 (Port 1238)**: Llama 3.1 8B Instruct (Q4_K_M)

Configure each instance to use the appropriate GPU via device mapping in LM Studio settings.

## Usage

### Interactive Mode

Launch an interactive chat session:

```bash
athena interactive
```

Or with custom configuration:

```bash
athena interactive --config my_config.yaml --log-level DEBUG
```

### Single Query Mode

Process a single query:

```bash
athena query "Explain the concept of recursion in programming"
```

### Health Check

Verify all model endpoints are operational:

```bash
athena health
```

### Python API

Use Athena programmatically:

```python
import asyncio
from athena import MetaOrchestrator, UserQuery, get_config

async def main():
    # Initialize
    config = get_config()
    orchestrator = MetaOrchestrator(config)

    # Create query
    query = UserQuery(
        query_id="unique-id",
        content="What is consciousness?"
    )

    # Process
    response = await orchestrator.process_query(query)

    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence:.2%}")
    print(f"Sources: {[s.value for s in response.sources]}")

    # Cleanup
    await orchestrator.close()

asyncio.run(main())
```

## Configuration

Configuration can be provided via environment variables (.env) or YAML files.

### Environment Variables

See `.env.example` for all available options. Key settings:

```bash
# Model Endpoints
ORCHESTRATOR_URL=http://localhost:1234/v1
REASONING_EXPERT_URL=http://localhost:1235/v1
# ... (see .env.example for complete list)

# Global Workspace Theory Parameters
GWT_ATTENTION_THRESHOLD=0.6
GWT_WORKSPACE_SIZE=5
GWT_BROADCAST_DECAY=0.9

# Expert Routing
ENABLE_PARALLEL_CONSULTATION=true
CONFIDENCE_THRESHOLD=0.7
```

### YAML Configuration

```yaml
orchestrator:
  url: http://localhost:1234/v1
  model_name: qwen2.5-14b-instruct
  gpu_id: 0
  temperature: 0.7

gwt:
  attention_threshold: 0.6
  workspace_size: 5
  broadcast_decay: 0.9
  enable_competition: true

# ... (see athena/config/default.yaml for complete example)
```

## Interactive Commands

When running in interactive mode, special commands are available:

- `/stats` - Show system statistics and expert usage
- `/health` - Run health checks on all endpoints
- `/workspace` - Show current Global Workspace state
- `/clear` - Clear conversation history and workspace
- `/help` - Show available commands
- `exit` or `quit` - End the session

## System Architecture

### Query Processing Flow

1. **User Query** → Orchestrator receives query
2. **Analysis** → Orchestrator classifies complexity and determines routing
3. **Expert Consultation** → Selected experts process query (parallel or sequential)
4. **Workspace Broadcast** → Expert responses compete for global workspace
5. **Synthesis** → Orchestrator merges expert outputs with conflict resolution
6. **Response** → Coherent response returned to user

### Routing Strategies

- **Direct**: Orchestrator handles simple queries without expert consultation
- **Single Expert**: Route to one specialized expert
- **Parallel**: Consult multiple experts simultaneously for multi-faceted problems
- **Sequential**: Chain experts for staged reasoning (e.g., analysis → implementation)

## Consciousness-Informed Design

Athena's architecture embodies principles from cognitive science:

- **Global Workspace Theory**: Information broadcasting and attention competition
- **Modular Mind**: Specialized cognitive faculties (expert models)
- **Executive Function**: Meta-cognitive coordination (orchestrator)
- **Working Memory**: Context maintenance (Memory Expert + conversation history)
- **Selective Attention**: Attention weighting and workspace competition

This creates an AI system whose knowledge architecture mirrors cognitive theories of mind rather than implementing monolithic transformer processing.

## Development

### Project Structure

```
athena/
├── core/              # Core system components
│   ├── orchestrator.py    # Meta-orchestrator implementation
│   ├── expert_base.py     # Expert model base class
│   └── workspace.py       # Global Workspace implementation
├── experts/           # Specialized expert models
│   ├── reasoning.py       # Reasoning specialist
│   ├── creative.py        # Creative synthesis specialist
│   ├── technical.py       # Technical implementation specialist
│   └── memory.py          # Memory and context specialist
├── communication/     # Inter-model communication
│   ├── schemas.py         # Message schemas (Pydantic models)
│   ├── protocol.py        # Communication protocol
│   └── lm_studio_client.py  # LM Studio API client
├── config/            # Configuration management
│   ├── settings.py        # Configuration classes
│   └── default.yaml       # Default configuration
└── cli.py             # Command-line interface
```

### Running Tests

```bash
pytest
# With coverage:
pytest --cov=athena --cov-report=html
```

### Code Quality

```bash
# Format code:
black athena/

# Lint:
ruff athena/

# Type checking:
mypy athena/
```

## Performance Considerations

### VRAM Usage

- **GPU 1 (7900 XT - 20GB)**: Qwen2.5 14B (~10GB) + context headroom
- **GPU 2 (7900 XT - 20GB)**: 4 expert models (~18GB total) + context headroom

### Optimization Tips

1. **Adjust quantization**: Use Q4_K_M for lower VRAM, Q5_K_M for better quality
2. **Tune context length**: Reduce `max_context_length` if running out of VRAM
3. **Disable parallel consultation**: Set `ENABLE_PARALLEL_CONSULTATION=false` for sequential processing (lower memory, slower)
4. **Workspace size**: Reduce `GWT_WORKSPACE_SIZE` to limit memory overhead

## Extending Athena

### Adding a New Expert

1. Create a new expert class in `athena/experts/`:

```python
from athena.core.expert_base import ExpertModel
from athena.communication.schemas import ExpertType

class MyExpert(ExpertModel):
    def build_system_prompt(self) -> str:
        return "You are a specialist in..."

    # Implement other required methods...
```

2. Add the expert type to `ExpertType` enum in `schemas.py`
3. Register in orchestrator's `_initialize_experts()`
4. Configure endpoint in settings

## Troubleshooting

### Common Issues

**Q: "Connection refused" errors**
A: Ensure LM Studio is running and models are loaded on the correct ports. Run `athena health` to check.

**Q: Out of VRAM errors**
A: Reduce quantization (Q4 instead of Q5), lower context length, or reduce number of active experts.

**Q: Slow response times**
A: Check GPU utilization, ensure models are on correct GPUs, consider disabling parallel consultation for sequential processing.

**Q: Inconsistent expert selection**
A: Adjust `GWT_ATTENTION_THRESHOLD` and routing parameters in configuration.

## Philosophical Foundations

Athena explores the intersection of AI architecture and consciousness studies:

- **How AI knows vs. how humans know**: Distributed cognitive processing mirroring modular theories of mind
- **Computational consciousness**: Implementing GWT's attention and broadcast mechanisms
- **Meta-cognitive awareness**: The orchestrator's ability to reason about its own reasoning process

This architecture serves both practical purposes (better multi-domain reasoning) and theoretical exploration (computational models of consciousness).

## Future Directions

- **Cloud integration**: Route complex queries to larger models (H100 via Hyperbolic)
- **Fine-tuning**: Specialize base models for expert roles
- **Meta-learning**: Orchestrator learns optimal routing strategies
- **Narrative specialist**: Additional expert for Project Exodus dialogue systems
- **Multi-turn planning**: Enhanced sequential reasoning chains

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

- Inspired by Global Workspace Theory (Baars, Franklin)
- Mixture of Experts architectures
- Cognitive science theories of modular mind
- The open-source LLM community

## Contact

For questions, issues, or discussions, please open an issue on GitHub.

---

**Built with consciousness-inspired architecture for advanced AI reasoning.**
