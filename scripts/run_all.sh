#!/bin/bash

# Run install.sh script
echo "Running install.sh script..."
bash install.sh

# Check the installation status
install_status=$?

# Check if installation was successful
if [ $install_status -eq 0 ]; then
    echo "Installation successful."
    echo "Running infer.sh script..."
    bash infer.sh
else
    echo "Installation failed. Exiting..."
fi

