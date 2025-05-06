import os
import argparse
import subprocess
import shutil

def run_textual_inversion(input_image=None, image_index=None, images_dir="./generated_images", 
                         config_path="configs/latent-diffusion/inversion_config.yaml", 
                         output_dir="./inversion_results", image_id=2,
                         prompt_file_path=None):
    """
    Run main_textual_inversion.py with the given input image or image index
    
    Args:
        input_image: Direct path to input image (optional if image_index is provided)
        image_index: Index of the image to process (optional if input_image is provided)
        images_dir: Directory containing generated images with pattern image_XXXX.png
        config_path: Path to config file
        output_dir: Directory to save results
        image_id: Image ID parameter for inversion
        prompt_file_path: Path to prompt file (optional if using index)
    """
    
    # Determine the input image path
    if input_image is None and image_index is not None:
        # Construct path from index
        input_image = os.path.join(images_dir, f"image_{int(image_index):04d}.png")
        print(f"Using image at: {input_image}")
    
    if input_image is None:
        raise ValueError("Either input_image or image_index must be provided")
    
    # Create unique output directory if using index
    if image_index is not None:
        output_dir = os.path.join(output_dir, f"image_{int(image_index):04d}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the input image exists
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"Input image not found: {input_image}")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Get absolute path to the input image
    image_abs_path = os.path.abspath(input_image)
    
    # Create prompt_file_path if not provided
    if prompt_file_path is None:
        # Set up unique prompt file path
        logs_forward_pass_dir = "logs_forward_pass"
        os.makedirs(logs_forward_pass_dir, exist_ok=True)
        
        # Create a unique prompt file name for this run
        if image_index is not None:
            prompt_file_name = f"prompt_file_{int(image_index):04d}.txt"
        else:
            prompt_file_name = "prompt_file.txt"
        
        prompt_file_path = os.path.join(logs_forward_pass_dir, prompt_file_name)
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)
    
    # Create a custom config file with the image path
    custom_config_path = os.path.join(output_dir, "custom_inversion_config_image_"+str(image_index)+".yaml")
    
    # Copy and modify the config file to include the image path
    with open(config_path, 'r') as original_config:
        config_content = original_config.read()
        
        # Replace the image path in the config file
        if "image_path:" in config_content:
            config_content = config_content.replace(
                'image_path: "./ldm/data/squirrel.png"', 
                f'image_path: "{image_abs_path}"'
            )
        
    with open(custom_config_path, 'w') as modified_config:
        modified_config.write(config_content)
    
    print(f"Created custom config at {custom_config_path}")
    
    # Prepare command for main_textual_inversion.py
    cmd = [
        "python", 
        "main_textual_inversion.py", 
        "--base", custom_config_path, 
        "--train", "True",
        "--prompt_file_path", prompt_file_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Log file for stdout
        log_path = os.path.join(output_dir, "textual_inversion.log")
        with open(log_path, 'w') as log_file:
            # Process and display stdout in real-time
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
        
        # Wait for the process to complete
        process.wait()
        
        # Check if the process was successful
        if process.returncode != 0:
            print(f"Error: main_textual_inversion.py failed with return code {process.returncode}")
            with open(os.path.join(output_dir, "error.log"), 'w') as error_file:
                error_file.write(process.stderr.read())
            return False, prompt_file_path
        
        print("Textual inversion completed successfully")
        
        # Check for generated prompt file
        if os.path.exists(prompt_file_path):
            # Copy the prompt file to the output directory
            output_prompt_file = os.path.join(output_dir, os.path.basename(prompt_file_path))
            shutil.copy(prompt_file_path, output_prompt_file)
            print(f"Copied generated prompts to {output_prompt_file}")
        else:
            print(f"Warning: Expected prompt file not found at {prompt_file_path}")
            # Create empty prompt file if it doesn't exist
            with open(prompt_file_path, 'w') as f:
                f.write("# Generated empty prompt file\n")
            print(f"Created empty prompt file at {prompt_file_path}")
            
        return True, prompt_file_path
        
    except Exception as e:
        print(f"Error running textual inversion: {e}")
        return False, prompt_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run textual inversion with specified input image")
    parser.add_argument("--input_image", help="Path to input image (optional if --image_index is used)")
    parser.add_argument("--image_index", type=int, help="Index of the image to process (optional if --input_image is used)")
    parser.add_argument("--images_dir", default="./generated_images", help="Directory containing generated images")
    parser.add_argument("--config_path", default="configs/latent-diffusion/inversion_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--output_dir", default="./inversion_results", help="Directory to save results")
    parser.add_argument("--image_id", type=int, default=2, help="Image ID parameter")
    parser.add_argument("--prompt_file_path", default=None, help="Path to prompt file")
    args = parser.parse_args()
    
    if args.input_image is None and args.image_index is None:
        parser.error("At least one of --input_image or --image_index must be specified")
    
    success, prompt_file = run_textual_inversion(
        input_image=args.input_image,
        image_index=args.image_index,
        images_dir=args.images_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        image_id=args.image_id,
        prompt_file_path=args.prompt_file_path
    )
    
    if success:
        print(f"Task completed successfully. Prompt file: {prompt_file}")
    else:
        print("Task failed.") 