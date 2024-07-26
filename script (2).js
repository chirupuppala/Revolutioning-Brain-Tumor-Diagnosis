document.addEventListener('DOMContentLoaded', () => {
    const flashes = document.querySelectorAll('.flashes li');
    flashes.forEach(flash => {
        setTimeout(() => {
            flash.style.display = 'none';
        }, 5000);
    });
});
