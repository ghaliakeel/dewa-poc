# DEWA POC Frontend

This repository contains the frontend code for a Proof of Concept (POC) developed for the Dubai Electricity & Water Authority (DEWA). The POC is built using ReactJS and is designed to work in conjunction with a backend developed using Flask for Low-Level Model (LLM) inference. Please note that you need to have Node.js installed to run the frontend.

## Getting Started

To get started with this code, follow these steps:

1. **Install Node.js**: Ensure that you have Node.js installed on your system. You can download and install it from [https://nodejs.org/](https://nodejs.org/).

2. **Clone this Repository**: Clone this repository to your local machine using Git:

    ```
    git clone <repository-url>
    ```

3. **Install Dependencies**: Navigate to the project directory and install the required dependencies using npm:

    ```
    cd dewa-poc
    npm install
    ```

4. **Start the Development Server**: Once the dependencies are installed, start the development server by running:

    ```
    npm start
    ```

    This will launch the frontend application, and you can access it in your web browser at [http://localhost:3000](http://localhost:3000).

## Important Note

Please note that the backend for this POC is developed using Flask for Low-Level Model (LLM) inference. To ensure the frontend communicates with the correct backend, make sure to update the URL configuration in the frontend code.

```javascript
// Update the backend URL in your code
const backendURL = 'http://your-backend-url';
