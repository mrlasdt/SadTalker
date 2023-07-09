#!/bin/bash
echo "Running download_models.sh script..."
bash scripts/download_models.sh

# Run install.sh script
echo "Running install.sh script..."
bash scripts/install.sh

# Check the installation status
install_status=$?

# Check if installation was successful
if [ $install_status -eq 0 ]; then
    echo "Installation successful."
    echo "Running infer.sh script..."
    bash scripts/infer.sh
else
    echo "Installation failed. Exiting..."
fi

