% part 1
% Load the PGM image
img = imread('path/to/image.pgm');

% Define the amount of noise to add (adjust this as needed)
noise_amount = 0.05;

% Add random Gaussian noise to the image
noisy_img = imnoise(img, 'gaussian', 0, noise_amount);

% Save the noisy image as a new PGM file
imwrite(noisy_img, 'path/to/noisy_image.pgm', 'pgm');



% part 2
% Load the noisy PGM image
noisy_img = imread('path/to/noisy_image.pgm');

% Apply median filter to remove Gaussian noise
denoised_img = medfilt2(noisy_img);

% Save the denoised image as a new PGM file
imwrite(denoised_img, 'path/to/denoised_image.pgm', 'pgm');



% part 3
% Load the image
img = imread('path/to/image.jpg');

% Convert the image to grayscale (if it's not already)
gray_img = rgb2gray(img);

% Perform histogram equalization
[eq_img, histogram] = histeq(gray_img);

% Save the equalized image as a PGM file
imwrite(eq_img, 'path/to/equalized_image.pgm', 'pgm');

% Display the original and equalized images side-by-side with their histograms
subplot(2, 2, 1), imshow(gray_img), title('Original Image');
subplot(2, 2, 2), imhist(gray_img), title('Original Image Histogram');
subplot(2, 2, 3), imshow(eq_img), title('Equalized Image');
subplot(2, 2, 4), bar(histogram), title('Equalized Image Histogram');
