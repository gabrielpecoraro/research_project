# README.md

# Traveling Salesman Problem (TSP) Project

This project implements a solution to the Traveling Salesman Problem (TSP) using reinforcement learning techniques. The goal is to find the shortest possible route that visits a set of cities and returns to the origin city.

## Project Structure

```
tsp-project
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── environment
│   │   ├── __init__.py
│   │   └── delivery.py
│   ├── agents
│   │   ├── __init__.py
│   │   └── delivery_q_agent.py
│   └── utils
│       ├── __init__.py
│       ├── plotting.py
│       └── helpers.py
├── tests
│   ├── __init__.py
│   ├── test_environment.py
│   └── test_agent.py
├── requirements.txt
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run the project, execute the `main.py` file:

```
python src/main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.