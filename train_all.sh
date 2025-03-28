if [ "$1" = "-O" ]; then
    python3 -O ./src/train.py &
    python3 -O ./src/momentum_experiments.py &
    python3 -O ./src/normalization.py &
    python3 -O ./src/deep_network.py &
else
    python3 ./src/train.py &
    python3 ./src/momentum_experiments.py &
    python3 ./src/normalization.py &
    python3 ./src/deep_network.py &
fi
wait
echo "All training done"
