#!/bin/bash

# ignore ipynb since their base64 images can include "todo"
ignored="$(basename $0) .ipynb __pycache__"

shopt -s globstar
code=0
for file in **/*; do
    ignore_this=false
    for pattern in $ignored; do
        if [[ "$file" = *"$pattern"* ]]; then
            ignore_this=true
            break
        fi
    done

    if [ -f "$file" -a $ignore_this == false ]; then
        # split into two commands for coloring
        grep -i todo "$file"
        if [ -n "$(grep -i todo "$file")" ]; then
            code=1
        fi
    fi
done

exit $code
