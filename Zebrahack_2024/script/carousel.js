const carouselImages = document.querySelector('.carousel-images');
const images = document.querySelectorAll('.carousel-images img');
const prevButton = document.querySelector('.carousel-btn.prev');
const nextButton = document.querySelector('.carousel-btn.next');

let index = 0;

function updateCarousel() {
    const width = images[0].clientWidth;
    carouselImages.style.transform = `translateX(${-index * width}px)`;
}

nextButton.addEventListener('click', () => {
    index = (index + 1) % images.length;
    updateCarousel();
});

prevButton.addEventListener('click', () => {
    index = (index - 1 + images.length) % images.length;
    updateCarousel();
});

window.addEventListener('resize', updateCarousel);
