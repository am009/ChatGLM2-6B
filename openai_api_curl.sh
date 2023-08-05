#!/bin/bash

# curl https://api.openai.com/v1/completions \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer None" \
# -d '{"model": "text-davinci-003", "prompt": "Say this is a test", "temperature": 0, "max_tokens": 7}'

curl -X POST "http://127.0.0.1:8086" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
