#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3 model. Please wait while downloading..."
ollama run ${LLAMA_MODEL}
if [ $? -eq 0 ]; then
    echo "🟢 Model initialization successful!"
else
    echo "❌ Model initialization failed!"
    exit 1
fi

READY_FILE="/tmp/ollama_pulled"

# Create the readiness marker file with the random name.
touch "$READY_FILE"
echo "Created readiness marker: $READY_FILE"


# Wait for Ollama process to finish.
wait $pid
