import requests

def retrieve_recipe_photo(recipe_id):
    url = 'https://api.food.com/external/v1/nlp/search'
    payload = {
        'contexts': [],
        'searchTerm': recipe_id,
        'pn': 1
    }

    try:
        # Perform the POST request
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()

            # Extract the primary photo URL from the first result if has_photo is 1
            if response_data['response']['results']:
                first_result = response_data['response']['results'][0]
                if first_result.get('has_photo') == '1':
                    return first_result.get('primary_photo_url')
                else:
                    return None
            else:
                return None
        else:
            return None
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return None



if __name__ == "__main__":
    search_term = "286971"
    result = retrieve_recipe_photo(search_term)
    print(result)