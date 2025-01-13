import cv2
from model import ChickenCounter
import os

def main():
    print("Starting chicken detection program...")
    
    # Initialize the counter
    try:
        print("Initializing ChickenCounter...")
        counter = ChickenCounter(confidence_threshold=0.5)
        print("ChickenCounter initialized successfully!")
        
        # Get the current directory
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        
        # List available files
        print("Files in current directory:")
        for file in os.listdir(current_dir):
            print(f"- {file}")
        
        # Replace this with one of your actual image files
        image_path = input("Enter the name of your image file (e.g., 'chicken.jpg'): ")
        
        if not os.path.exists(image_path):
            print(f"Error: The file '{image_path}' does not exist!")
            return
            
        print(f"Processing image: {image_path}")
        
        # Get count and visualized image
        count, annotated_image = counter.count_chickens(image_path, visualize=True)
        print(f"Number of chickens detected: {count}")
        
        # Save the annotated image
        output_path = "output_annotated.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()