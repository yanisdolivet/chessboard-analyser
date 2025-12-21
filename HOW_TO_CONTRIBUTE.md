# How to Contribute

Thank you for your interest in contributing to the Chess Position Analyzer project. This guide will help you get started with contributing to the codebase.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy library
- Git for version control

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chessboard-analyser.git
   cd chessboard-analyser
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused on a single responsibility

### Testing Your Changes

Before submitting your changes:

1. Test the neural network training:
   ```bash
   make train
   ```

2. Test the analyzer:
   ```bash
   make predict
   ```

3. Verify your changes don't break existing functionality

### Commit Guidelines

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable
- Keep commits focused on a single change

Example:
```bash
git commit -m "Add dropout regularization to Layer class"
```

## Submitting Your Contribution

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub:
   - Provide a clear title and description
   - Explain what changes you made and why
   - Reference any related issues
   - Include screenshots or examples if applicable

3. Wait for review:
   - Address any feedback from maintainers
   - Make requested changes in your branch
   - Push updates to your PR branch

## Types of Contributions

### Bug Fixes

- Check existing issues before creating a new one
- Provide steps to reproduce the bug
- Include error messages and logs
- Test your fix thoroughly

### New Features

- Open an issue to discuss the feature before implementing
- Ensure the feature aligns with project goals
- Update documentation as needed
- Add examples showing how to use the feature

### Documentation

- Fix typos or unclear explanations
- Add examples or clarify existing ones
- Update outdated information
- Improve code comments

### Performance Improvements

- Benchmark your improvements
- Document performance gains
- Ensure accuracy is not compromised
- Test on different dataset sizes

## Project Structure

Understanding the codebase:

```
src/
├── my_torch/           # Neural network library
│   ├── Network.py      # Network management and training
│   ├── Layer.py        # Layer implementation
│   └── DataAnalysis.py # Metrics and evaluation
├── analyzer/           # Chess-specific components
│   ├── FENParser.py    # FEN string parsing
│   └── ModelLoader.py  # Binary model loading
└── my_torch_analyzer.py # Main application

tools/
├── generator/          # Network generation tools
└── balance_dataset.py  # Dataset utilities

config/
└── basic_network.json  # Network configuration
```

## Areas for Contribution

Current areas where contributions are welcome:

1. **Dataset Generation**: Tools to create larger, more diverse datasets
2. **Performance Optimization**: Speed improvements for training or inference
3. **Visualization**: Better plots and analysis tools
4. **Documentation**: Improve guides and examples
5. **Testing**: Add unit tests and integration tests
6. **Architecture**: Experiment with different network topologies
7. **Features**: Additional chess position analysis capabilities

## Questions or Issues?

- Check existing issues on GitHub
- Review documentation in `docs/` folder
- Open a new issue for questions or problems
- Be respectful and constructive in discussions

## Code of Conduct

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences
- Accept responsibility for mistakes

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to the Chess Position Analyzer project!
