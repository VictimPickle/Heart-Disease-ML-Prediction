#!/usr/bin/env python3
"""
Script to upload visualization images to GitHub repository.
Run this script after generating the visualizations from heart_disease_analysis.py
"""

import os
import base64
from github import Github
import sys

def upload_images_to_github(token, repo_name, image_folder='.'):
    """
    Upload visualization images to GitHub repository.
    
    Args:
        token: GitHub personal access token
        repo_name: Repository name (e.g., 'VictimPickle/Heart-Disease-ML-Prediction')
        image_folder: Folder containing the images (default: current directory)
    """
    # Image mapping: local filename -> GitHub path
    image_files = {
        '01_correlation_heatmap.png': 'images/correlation_heatmap.jpg',
        'correlation_heatmap.jpg': 'images/correlation_heatmap.jpg',
        '02_feature_importance.png': 'images/feature_importance.jpg',
        '03_model_metrics.png': 'images/model_metrics.jpg',
        '05_false_negatives.png': 'images/false_negatives.jpg',
        '06_decision_tree.png': 'images/decision_tree.jpg',
        'decision_tree.jpg': 'images/decision_tree.jpg',
    }
    
    try:
        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(repo_name)
        
        print(f"Uploading images to {repo_name}...")
        
        # Upload each image
        for local_file, github_path in image_files.items():
            local_path = os.path.join(image_folder, local_file)
            
            if not os.path.exists(local_path):
                print(f"Skipping {local_file} (not found)")
                continue
            
            # Read image file as binary
            with open(local_path, 'rb') as f:
                content = f.read()
            
            # Convert to base64
            encoded_content = base64.b64encode(content).decode('utf-8')
            
            try:
                # Check if file exists
                try:
                    existing_file = repo.get_contents(github_path)
                    # Update existing file
                    repo.update_file(
                        github_path,
                        f"Update {os.path.basename(github_path)}",
                        encoded_content,
                        existing_file.sha
                    )
                    print(f"✓ Updated: {github_path}")
                except:
                    # Create new file
                    repo.create_file(
                        github_path,
                        f"Add {os.path.basename(github_path)}",
                        encoded_content
                    )
                    print(f"✓ Created: {github_path}")
            except Exception as e:
                print(f"✗ Failed to upload {local_file}: {str(e)}")
        
        print("\nUpload complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    # Instructions
    print("="*60)
    print("GitHub Image Upload Script")
    print("="*60)
    print()
    print("To use this script:")
    print("1. Install PyGithub: pip install PyGithub")
    print("2. Create a GitHub token: https://github.com/settings/tokens")
    print("3. Run: python upload_images.py")
    print()
    print("="*60)
    print()
    
    # Get token from user
    token = input("Enter your GitHub Personal Access Token: ").strip()
    
    if not token:
        print("Error: Token is required!")
        sys.exit(1)
    
    repo_name = "VictimPickle/Heart-Disease-ML-Prediction"
    image_folder = "."
    
    upload_images_to_github(token, repo_name, image_folder)
