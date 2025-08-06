# Manual Annotations

<img width="1205" height="830" alt="Manual Annotation Interface" src="../assets/AcceptRejcetAnnotate.png" />

Manual annotation is the primary purpose of this tool. After generating question-answer pairs, manually annotating all videos becomes essential for quality assurance and dataset refinement.

## Workflow Overview

### 1. Loading Videos

- Click **Browse Directory** to navigate to your video folder
- Select the desired video from the file list by either:
- Using **Load Selected** button
- Double-clicking on the video file
- The video will load in the main frame

### 2. Analysis Mode Selection

Once the video is loaded, you can switch between two analysis modes:

- **Video Analysis**: Standard playback with full video controls
- **Frame Analysis**: Frame-by-frame examination for detailed annotation

### 3. Loading Question Data

- Load the CSV file containing your questions using the file browser
- Questions will be displayed with the following columns:
- **Category**: Question classification
- **Question**: The actual question text
- **Answer**: Corresponding answer
- **Status**: Current annotation status

> **Note**: The questions section remains locked until the video playback is complete, ensuring you review the entire video content before beginning annotations.

### 4. Annotation Process

1. **Review**: Read each question carefully while referring to the video content
2. **Accept**: Click on questions that are accurate and relevant (no additional action required)
3. **Reject & Edit**: For problematic questions:

- Select the question as rejected
- Double-click to open the edit dialog
- Modify the question or answer as needed

4. **Reset**: Use the reset button if you make a mistake during annotation

### 5. Saving Progress

- **Save Template**: Store your progress temporarily before completing the full annotation
- **Export**: Generate the complete chat template once all annotations are finished

## Video Demonstration

[![Manual Annotation Demo](https://img.youtube.com/vi/rTfHd3GMrf8/0.jpg)](https://www.youtube.com/watch?v=rTfHd3GMrf8)

_Click the image above to watch a detailed demonstration of the manual annotation process._
