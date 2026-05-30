#!/bin/bash

shopt -s globstar
code=0
for file in **/*; do
    if [ -f "$file" ]; then
        # split into two commands for coloring
        grep -i todo "$file"
        if [ -n "$(grep -i todo "$file")" ]; then
            code=1
        fi
    fi
done

exit $code
