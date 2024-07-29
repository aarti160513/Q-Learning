## Q-Learning

This project implements the Q-Learning algorithm to solve the Frozen Lake environment from the gymnasium library. The environment is a grid where an agent must navigate from a starting point to a goal while avoiding holes. The agent is rewarded only after ataining the goal

# Set-up

Follow the instructions below to create a virtual environment specific for this projest

1. install pipenv

```pwsh
   pip install pipenv
```

2. clone the repo

```pwsh
git clone https://github.com/aarti160513/Qlearning.git .
```

3. Create and activate the virtual environment

```pwsh
# skip the command below if you don't need virtual environment within in your local directory
$Env:PIPENV_VENV_IN_PROJECT=1

# create and activate the virtual environment
pipenv shell
pipenv sync
```

# FrozenLake Run

update the settings in settins.toml file to test different permutations and run the code to see agent traverse the lake.

```pwsh
python main.py
```

Sample response on my machine

(Right)<br>
SFHFFH<br>
FFFFFH<br>
FHFFFF<br>
HFFFHH<br>
FFFFFF<br>
FHFFF**G**
