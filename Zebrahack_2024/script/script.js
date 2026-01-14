


//buton light dark
// Selectează butonul și corpul paginii
const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Verifică dacă există preferințe salvate în localStorage
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    body.classList.add(savedTheme);
    updateButtonText(savedTheme);
}

// Adaugă evenimentul de click pe buton
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('dark-mode')) {
        // Comută la Light Mode
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        localStorage.setItem('theme', 'light-mode'); // Salvează preferința
        updateButtonText('light-mode');
    } else {
        // Comută la Dark Mode
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark-mode'); // Salvează preferința
        updateButtonText('dark-mode');
    }
});

function updateButtonText(currentTheme) {
    if (currentTheme === 'dark-mode') {
        modeToggle.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-lightbulb" viewBox="0 0 16 16">
  <path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13a.5.5 0 0 1 0 1 .5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1 0-1 .5.5 0 0 1 0-1 .5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m6-5a5 5 0 0 0-3.479 8.592c.263.254.514.564.676.941L5.83 12h4.342l.632-1.467c.162-.377.413-.687.676-.941A5 5 0 0 0 8 1"/>
</svg>`;
    } else {
        modeToggle.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-lightbulb-fill" viewBox="0 0 16 16">
  <path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13h-5a.5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m3 8.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1-.5-.5"/>
</svg>`;
    }
}

// Selectează butonul și corpul paginii
const modeToggle1 = document.getElementById('modeToggle1');

// Verifică dacă există preferințe salvate în localStorage
const savedTheme1 = localStorage.getItem('theme');
if (savedTheme) {
    body.classList.add(savedTheme);
    updateButtonText(savedTheme);
}

// Adaugă evenimentul de click pe buton
modeToggle1.addEventListener('click', () => {
    if (body.classList.contains('dark-mode')) {
        // Comută la Light Mode
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        localStorage.setItem('theme', 'light-mode'); // Salvează preferința
        updateButtonText('light-mode');
    } else {
        // Comută la Dark Mode
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark-mode'); // Salvează preferința
        updateButtonText('dark-mode');
    }
});

function updateButtonText(currentTheme) {
    if (currentTheme === 'dark-mode') {
        modeToggle.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-lightbulb" viewBox="0 0 16 16">
  <path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13a.5.5 0 0 1 0 1 .5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1 0-1 .5.5 0 0 1 0-1 .5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m6-5a5 5 0 0 0-3.479 8.592c.263.254.514.564.676.941L5.83 12h4.342l.632-1.467c.162-.377.413-.687.676-.941A5 5 0 0 0 8 1"/>
</svg>`;
    } else {
        modeToggle.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-lightbulb-fill" viewBox="0 0 16 16">
  <path d="M2 6a6 6 0 1 1 10.174 4.31c-.203.196-.359.4-.453.619l-.762 1.769A.5.5 0 0 1 10.5 13h-5a.5.5 0 0 1-.46-.302l-.761-1.77a2 2 0 0 0-.453-.618A5.98 5.98 0 0 1 2 6m3 8.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1l-.224.447a1 1 0 0 1-.894.553H6.618a1 1 0 0 1-.894-.553L5.5 15a.5.5 0 0 1-.5-.5"/>
</svg>`;
    }
}

function resetEmail() {
    document.getElementById("email").value = "";
}

function handleSubmit(event) {
    event.preventDefault(); 

    const email = document.getElementById("email").value; 
    const subscribe = document.getElementById("subs");

    if (email) {

        subscribe.innerHTML = `Mulțumim de abonare, <strong>${email}</strong>`;
        setTimeout(() => {
            subscribe.style.display = "none";
        }, 2000);
    }
}

function openWhatsApp() {
    window.open("https://chat.whatsapp.com/EGHyZ5S7JzCKPbUoBmxBiZ", "_blank");
}