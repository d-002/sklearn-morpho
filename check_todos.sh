#!/bin/bash

# split word to avoid grepping this very file
word=$(echo to)$(echo do)

shopt -s globstar
code=0
for file in **/*; do
    if [ -f "$file" ]; then
        # split into two commands for coloring
        grep -i "$word" "$file"
        if [ -n "$(grep -i "$word" "$file")" ]; then
            code=1
        fi
    fi
done

exit $code
