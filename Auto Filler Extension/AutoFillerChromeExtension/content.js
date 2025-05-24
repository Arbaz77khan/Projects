// Content script to autofill application forms

// Example key-value pairs (to be fetched from backend later)
let autofillData = {
    "First name": "Arbaz",
    "Last name": "Khan",
    "Email": "arbazkhan772001@gmail.com"
};
  
  // Function to autofill form fields
function autofillForm() {
    const inputs = document.querySelectorAll('input, textarea');

    inputs.forEach(input => {
        let fieldName = input.getAttribute('name') || input.getAttribute('id');
        if (autofillData[fieldName]) {
        input.value = autofillData[fieldName];
        }
    });
}
  
  // Run the autofill function when the page loads
window.onload = function() {
    autofillForm();
};
  