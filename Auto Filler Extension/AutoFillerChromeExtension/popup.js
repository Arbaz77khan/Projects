document.getElementById('saveBtn').addEventListener('click', function() {
    let name = document.getElementById('name').value;
    let email = document.getElementById('email').value;
    let salary = document.getElementById('salary').value;
  
    // Save key-value pairs in local storage (or send to backend via API)
    chrome.storage.sync.set({ name: name, email: email, salary: salary }, function() {
      console.log('Values saved:', { name, email, salary });
    });
  });
  