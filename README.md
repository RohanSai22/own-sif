# 🔥 Prometheus 2.0 - The Observable Darwinian Gödeli Machine

An advanced self-improving AI agent that evolves through code mutation and performance feedback, with complete real-time observability.

## 🧬 Philosophy

Prometheus 2.0 embodies the concept of "Evolution, Televised" - a fully observable self-improving AI system where every thought, action, and evolutionary step is transparent. Unlike black-box AI systems, Prometheus provides real-time insights into the agent's decision-making process, tool usage, and self-modification attempts.

The system implements **Darwinian Gödeli Machine (DGM)** principles:

- **Self-Improvement**: The agent can modify its own source code
- **Natural Selection**: Better-performing agents become parents of the next generation
- **Mutation**: Code changes introduce variation for evolution
- **Evaluation**: Performance on SWE-bench determines fitness
- **Observability**: Every step is logged and displayed in real-time

## 🏗️ Architecture

```
prometheus_dgm/
├── main.py                    # Main orchestrator
├── config.py                  # Configuration and settings
├── requirements.txt           # Dependencies
├── .env.example              # Environment variables template
│
├── framework/                 # Core framework
│   ├── tui.py                # Terminal User Interface
│   └── mutator.py            # Code mutation system
│
├── agent/                     # Agent implementation
│   ├── agent_core.py         # Core agent logic
│   └── prompts.py            # LLM prompts and templates
│
├── llm_provider/             # LLM integration
│   └── unified_client.py     # Multi-provider LLM client
│
├── tools/                    # Agent tools
│   ├── base_tools.py         # Core tool implementations
│   ├── tool_manager.py       # Dynamic tool management
│   └── generated_tools/      # Auto-generated tools
│
├── evaluation/               # Evaluation system
│   ├── swe_bench_harness.py  # SWE-bench integration
│   └── sandbox.py            # Docker security sandbox
│
└── archive/                  # Agent generations
    └── agent_archive.py      # Generation management
```

## 🚀 Key Innovations

### 1. **Live TUI Dashboard**

Real-time terminal interface displaying:

- **Agent Inner Monologue**: Stream of agent thoughts
- **Live Action Log**: Real-time tool usage and results
- **Docker Evaluation Log**: Test execution output
- **Generational Performance**: Evolution tracking table
- **System Status**: Current iteration, scores, and progress

### 2. **API-Free Autonomous Operation**

- **Web Search**: Uses DuckDuckGo (no API key required)
- **Web Scraping**: BeautifulSoup4 for content extraction
- **Self-Contained**: No external API dependencies
- **Cost-Free**: No usage fees or rate limits

### 3. **Robust Self-Mutation System**

- **JSON Patch Format**: Structured code modifications
- **Syntax Validation**: Ensures code remains valid
- **Backup System**: Automatic rollback on errors
- **Safe Application**: Atomic code changes

### 4. **Secure Docker Evaluation**

- **Isolated Execution**: Each test runs in a fresh container
- **Automatic Cleanup**: No resource leaks
- **Streaming Output**: Real-time test results
- **Reproducible**: Consistent evaluation environment

### 5. **Dynamic Tool Creation**

- **Self-Expanding Toolkit**: Agent creates new tools as needed
- **Hot Loading**: Tools available immediately after creation
- **Dependency Management**: Automatic package installation
- **Usage Tracking**: Tool effectiveness monitoring

## 📋 Prerequisites

### Required Software

- **Python 3.11+**
- **Docker** (must be installed and running)
- **Git**

### Required API Keys (at least one)

- **Groq API Key** (recommended - fast and free tier)
- **OpenAI API Key** (GPT models)
- **Google Gemini API Key** (Gemini models)

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd prometheus_dgm
```

2. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:

```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Verify Docker installation**:

```bash
docker --version
docker run hello-world
```

## ⚙️ Configuration

Edit `.env` file with your API keys:

```env
# At least one API key is required
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional settings
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### System Configuration

Key settings in `config.py`:

```python
# Evolution parameters
max_iterations = 100
population_size = 5
mutation_rate = 0.3

# Evaluation settings
swe_bench_timeout = 300  # 5 minutes per task
max_concurrent_evaluations = 3

# Performance thresholds
score_improvement_threshold = 0.05  # 5% minimum improvement
stagnation_limit = 10  # iterations without improvement
```

## 🎮 Usage

### Basic Usage

Run Prometheus 2.0:

```bash
python main.py
```

This launches the Terminal User Interface and begins the evolution process.

### TUI Interface Guide

The TUI is divided into several panels:

1. **Header**: System status, iteration count, best score
2. **Agent Inner Monologue**: Real-time agent thoughts
3. **Live Action Log**: Tool calls and results
4. **Docker Evaluation Log**: Test execution output
5. **Current Status**: Current task and performance metrics
6. **Generational Performance**: Evolution history table

### Keyboard Controls

- **Ctrl+C**: Graceful shutdown
- **Ctrl+Z**: Force terminate (not recommended)

### Evolution Process

1. **Initialization**: Load or create genesis agent
2. **Evaluation**: Run agent on SWE-bench tasks
3. **Performance Analysis**: Calculate scores and improvements
4. **Self-Reflection**: Agent analyzes its performance
5. **Mutation**: Apply code changes to create new generation
6. **Archival**: Store generation for future reference
7. **Repeat**: Continue evolution loop

## 🔧 Advanced Usage

### Custom Tool Creation

The agent can create new tools dynamically. Tools are stored in `tools/generated_tools/` and automatically loaded.

### Manual Mutation

You can manually trigger mutations by modifying the agent's source code and restarting.

### Archive Management

Access archived generations:

```python
from archive.agent_archive import AgentArchive

archive = AgentArchive(".")
stats = archive.get_generation_stats()
best_agent = archive.get_best_agent()
```

### Docker Customization

Modify Docker settings in `config.py`:

```python
docker_image_base = "python:3.11-slim"
swe_bench_timeout = 300
```

## 📊 Monitoring and Analysis

### Performance Metrics

The system tracks:

- **Success Rate**: Percentage of tasks completed successfully
- **Average Score**: Mean performance across tasks
- **Execution Time**: Speed of problem-solving
- **Tool Usage**: Effectiveness of available tools
- **Generational Improvement**: Evolution progress

### Log Files

- **prometheus.log**: Detailed system logs
- **archive/evaluation_results/**: SWE-bench results
- **archive/generations/**: Agent generations metadata
- **archive/backups/**: Code mutation backups

### TUI Data Export

The TUI provides real-time data that can be exported for analysis:

```python
from framework.tui import tui
summary = tui.get_summary()
```

## 🐛 Troubleshooting

### Common Issues

1. **"No LLM providers are working"**

   - Check API keys in `.env` file
   - Verify internet connection
   - Try different provider

2. **"Docker is not available"**

   - Install Docker Desktop
   - Start Docker daemon
   - Check Docker permissions

3. **"Failed to load SWE-bench dataset"**

   - Check internet connection
   - Verify Hugging Face datasets access
   - Try reducing batch size

4. **"Mutation failed"**
   - Check code syntax in mutations
   - Review mutation JSON format
   - Check file permissions

### Debug Mode

Enable debug logging:

```env
LOG_LEVEL=DEBUG
DEBUG_MODE=true
```

### Performance Issues

If the system is slow:

1. Reduce `max_concurrent_evaluations`
2. Decrease `swe_bench_timeout`
3. Limit `max_tasks_per_iteration`
4. Use faster LLM models (Groq recommended)

## 🧪 Testing

Run component tests:

```bash
# Test LLM connectivity
python llm_provider/unified_client.py

# Test tool system
python tools/tool_manager.py

# Test Docker sandbox
python evaluation/sandbox.py

# Test TUI (interactive)
python framework/tui.py
```

## 🔬 Research Applications

Prometheus 2.0 is designed for AI research in:

- **Self-Improving AI**: Study autonomous code modification
- **Agent Evolution**: Observe AI capability development
- **Tool Usage**: Analyze dynamic skill acquisition
- **Performance Optimization**: Track learning efficiency
- **Transparency**: Understand AI decision-making

### Data Collection

The system generates rich datasets for research:

- Agent conversation logs
- Performance time series
- Code evolution diffs
- Tool usage patterns
- Evaluation outcomes

## 🛡️ Security Considerations

### Docker Isolation

All code execution happens in isolated Docker containers that are automatically destroyed after each evaluation.

### Code Validation

The mutation system validates all code changes before application and maintains backups for rollback.

### Network Access

Agents only have access to:

- Web search (DuckDuckGo)
- Git repositories (read-only)
- SWE-bench datasets

## 📈 Performance Optimization

### LLM Selection

For best performance:

1. **Groq** (fastest, free tier available)
2. **OpenAI GPT-4** (highest quality)
3. **Gemini Pro** (good balance)

### Resource Management

Monitor system resources:

- Docker container memory usage
- Disk space for archives
- Network bandwidth for datasets

### Batch Size Tuning

Optimize `max_concurrent_evaluations` based on:

- Available CPU cores
- Memory capacity
- Docker performance

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black isort mypy
   ```
4. Run tests before submitting

### Code Style

- Use Black for formatting
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings

### Adding New Features

1. **New Tools**: Add to `tools/base_tools.py`
2. **New Evaluations**: Extend `evaluation/swe_bench_harness.py`
3. **New Mutations**: Enhance `framework/mutator.py`
4. **New Visualizations**: Modify `framework/tui.py`

## 📜 License

[Add your license here]

## 🙏 Acknowledgments

- **SWE-bench**: For providing the evaluation benchmark
- **Hugging Face**: For dataset hosting
- **Rich/Textual**: For beautiful terminal interfaces
- **Docker**: For secure sandboxing
- **DuckDuckGo**: For API-free search

## 📞 Support

For issues, questions, or contributions:

- Open GitHub issues for bugs
- Use discussions for questions
- Submit PRs for contributions

---

**Prometheus 2.0** - Watching AI evolve, one generation at a time. 🔥🧬
