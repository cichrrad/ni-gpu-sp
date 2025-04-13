#!/bin/bash

# Default parameters
count=10
wmin=100
wmax=15000
hmin=100
hmax=15000
output_dir="downloaded_images"

# Function to display usage/help
usage() {
  echo "Usage: $0 [-c number_of_images] [-wmin minimum_width] [-wmax maximum_width] [-hmin minimum_height] [-hmax maximum_height] [-o output_directory]"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--count)
            count="$2"
            shift; shift;;
        -wmin)
            wmin="$2"
            shift; shift;;
        -wmax)
            wmax="$2"
            shift; shift;;
        -hmin)
            hmin="$2"
            shift; shift;;
        -hmax)
            hmax="$2"
            shift; shift;;
        -o|--output)
            output_dir="$2"
            shift; shift;;
        -h|--help)
            usage;;
        *)
            echo "Unknown option: $1"
            usage;;
    esac
done

# Ensure the output directory exists
mkdir -p "$output_dir"

echo "Attempting to fetch $count valid images with widths between $wmin and $wmax and heights between $hmin and $hmax."

valid_count=0
attempts=0

# Loop until we get the requested number of valid images
while [ "$valid_count" -lt "$count" ]; do
    attempts=$((attempts+1))
    
    # Generate random width and height within the specified ranges
    width=$(( RANDOM % (wmax - wmin + 1) + wmin ))
    height=$(( RANDOM % (hmax - hmin + 1) + hmin ))
    
    # Construct the initial Picsum URL
    url="https://picsum.photos/${width}/${height}"
    
    # Temporary filename for download (overwritten on each iteration)
    temp_file="$output_dir/tmp_image.jpg"
    
    # Use curl to download image, following redirects (-L)
    # The -w option with %{url_effective} retrieves the final URL.
    final_url=$(curl -Ls -o "$temp_file" -w "%{url_effective}" "$url")
    
    # Check if the final URL is the same as the requested URL. If yes, it's deemed invalid.
    if [ "$final_url" = "$url" ]; then
        echo "Attempt $attempts: Invalid image size ${width}x${height} (no redirection). Skipping..."
        rm -f "$temp_file"
        continue
    fi

    # Construct a unique filename based on the valid_count (starting at 1) and dimensions.
    filename="$output_dir/image_$((valid_count+1))_${width}x${height}.jpg"
    mv "$temp_file" "$filename"
    
    echo "Downloaded valid image $((valid_count+1)) (attempt $attempts):"
    echo "  Requested URL: $url"
    echo "  Final URL:     $final_url"
    echo "  Saved as:      $filename"
    
    valid_count=$((valid_count+1))
done

echo "Download completed. $valid_count valid images saved in $output_dir after $attempts attempts."

