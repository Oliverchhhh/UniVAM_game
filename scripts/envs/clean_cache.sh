#!/bin/bash

rm -rf ckpts/debug logs/debug

ROOT_DIR="logs"
TOPK=1

while getopts "r:k:" opt; do
    case $opt in
        r) ROOT_DIR="$OPTARG" ;;
        k) TOPK="$OPTARG" ;;
        *)
            echo "Usage: $0 -r <rootdir> -k <topk>"
            exit 1
            ;;
    esac
done

if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ]; then
    echo "Error: rootdir not specified or not exist."
    exit 1
fi

echo "ROOT_DIR = $ROOT_DIR"
echo "TOPK = $TOPK"
echo "===================================="

all_dirs=$(find "$ROOT_DIR" -type d)

while IFS= read -r dir; do
    subitems=()
    while IFS= read -r item; do
        subitems+=("$item")
    done < <(ls -1 "$dir" 2>/dev/null)

    numeric_items=()
    for base in "${subitems[@]}"; do
        if [[ "$base" =~ ^[0-9]+$ ]]; then
            numeric_items+=("$dir/$base")
        fi
    done

    if [ ${#numeric_items[@]} -le $TOPK ]; then
        continue
    fi

    echo "Processing: $dir"

    # top-k
    keep_items=$(printf "%s\n" "${numeric_items[@]}" | sort -V | tail -n "$TOPK")

    echo "  -> Keeping:"
    printf "     %s\n" $keep_items

    for item in "${numeric_items[@]}"; do
        if ! printf "%s\n" "$keep_items" | grep -q "$item"; then
            echo "  -> Removing: $item"
            rm -rf "$item"
        fi
    done

done <<< "$all_dirs"

echo "Done."
