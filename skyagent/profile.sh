
# Some great aliases from stas00

## default pytest command
alias pyt="pytest --disable-warnings --instafail -rA"

# show me all the available tests
alias pytc="pytest --disable-warnings --collect-only -q"


## ls aliases

# this is the most used alias
function l()  { ls -lNhF   --color=always "${@-.}" |more; }

# same as l but sort by latest
function lt() { ls -lNhtF  --color=always "${@-.}" |more; }

# same as l but include dot files
function la() { ls -lNhaF  --color=always "${@-.}" |more; }


# My git aliases 


alias ga="git add"
alias gcm="git commit -s -m"
alias gau="git add -u"

# nvidia-smi
alias wn='watch -n 1 nvidia-smi'
alias wnm='nvidia-smi --query-gpu=timestamp,utilization.memory,memory.used --format=csv -l 1'

# Uv aliases 
alias uvt="uv run --isolated --extra dev"
alias uvi="uv run --isolated"