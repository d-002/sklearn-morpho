#!/bin/sh

pkg_name="$(basename -- "$(pwd)")"
files=("PKGBUILD" ".SRCINFO")

if [ -e "python-$pkg_name" ]; then
    echo "Error: aur repo already cloned, please save your work and remove"
    exit 1
fi

updpkgsums
makepkg --printsrcinfo > .SRCINFO

git clone "ssh://aur@aur.archlinux.org/python-$pkg_name.git"
cd "python-$pkg_name"

for file in "${files[@]}"; do
    cp "../$file" .
done
