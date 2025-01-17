const API_URL = 'http://127.0.0.1:5000';  // Replace with your actual Flask server URL

// Function to fetch recipes from the backend (GET request)
export const fetchRecipes = async () => {
  const endpoint = '/recipes';  // Replace with your actual endpoint
  const fullURL = `${API_URL}${endpoint}`;

  try {
    const response = await fetch(fullURL);  // Sending GET request
    if (!response.ok) {
      throw new Error('Failed to fetch recipes');
    }
    const data = await response.json();  // Parse JSON response
    return data;
  } catch (error) {
    console.error('Error fetching recipes:', error);
    return null;  // Handle error case
  }
};

// Function to generate a recipe based on category (POST request)
export const generateRecipe = async (category) => {
  const endpoint = '/generate-recipe';  // Replace with your actual endpoint
  const fullURL = `${API_URL}${endpoint}`;

  const requestData = { category };  // Example of sending category data to generate recipe

  try {
    const response = await fetch(fullURL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),  // Send the category as JSON
    });

    if (!response.ok) {
      throw new Error('Failed to generate recipe');
    }
    const data = await response.json();  // Parse JSON response
    return data;
  } catch (error) {
    console.error('Error generating recipe:', error);
    return null;  // Handle error case
  }
};