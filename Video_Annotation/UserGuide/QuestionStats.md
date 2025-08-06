# Question Statisitics

<img width="1205" height="830" alt="image" src="../assets/Question_Stats.png" />

A comprehensive tool for video analysis, Q&A annotation, and machine learning dataset preparation with support for multiple video language models.

## Question Statistics

Interactive analytics dashboard providing comprehensive insights into your Q&A dataset through a 4-panel layout.

### Dashboard Overview

| Panel            | Content                  | Visualization                             |
| ---------------- | ------------------------ | ----------------------------------------- |
| **Top Left**     | Category Distribution    | Interactive pie chart with percentages    |
| **Top Right**    | Total Statistics Summary | Key metrics and counts                    |
| **Bottom Left**  | Objects in Answers       | Bar chart of detected objects             |
| **Bottom Right** | Question Types           | Horizontal bar chart by question patterns |

### Usage

1. Load a CSV file containing Q&A data
2. Click **\"Question Statistics\"** in the right panel
3. View interactive analytics in the popup window

### Key Metrics Tracked

#### Category Distribution

- Visual breakdown of question categories
- Percentage-based pie chart with legend
- Color-coded segments for identification

#### Dataset Statistics

- Total Q&A pairs count
- Unique video files
- Average Q&A per video
- Number of question categories

#### Object Detection in Answers

Automatically detects and counts mentions of:

- 🚗 Cars (car, vehicle, automobile, auto)
- 🚶 Pedestrians (pedestrian, people, person, walker)
- 🚲 Bicycles (bicycle, bike, cyclist)
- 🚛 Trucks (truck, lorry)
- 🚌 Buses (bus)
- 🏍️ Motorcycles (motorcycle, motorbike)
- 🚦 Traffic Lights (traffic light, signal)
- 🏢 Buildings (building, structure)
- 🛣️ Roads (road, street, highway)

#### Question Type Analysis

Categorizes questions by starting patterns:

- **What** - Information/explanation questions
- **Who** - Questions about people/entities
- **When** - Time/timing questions
- **Where** - Location/place questions
- **Why** - Reason/cause questions
- **How** - Method/process questions
- **Which** - Selection/choice questions
- **Can/Could** - Ability/possibility questions
- **Do/Does** - Action/state questions
