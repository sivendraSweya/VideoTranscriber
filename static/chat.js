 document.getElementById('chat-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const userInput = document.getElementById('user_input').value;

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput }) // ✅ Important!
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById('response').innerHTML = data;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
 
