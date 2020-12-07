#ifndef PANORAMIC_H
#define PANORAMIC_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <memory>

using namespace cv;
using namespace std;

// Define panoramic class structure
class Panoramic
{
	public:

        /**
        * @brief Load images from given path and number
        * @param path - images path
        * @param n    - number of images
        * @param show - DEFAULT FALSE -> if TRUE show loaded images
        * @return vector<Mat> that contains all the loaded images
        **/
        static vector<Mat> load_images(string path, int n, bool show = false);


        /**
        * @brief Project given images to cylindrical surface and converto to grey-scale or color
        * @param input    - vector that contains all images
        * @param output   - destination vector in which save projected images
        * @param half_fov - half-Field_of_View parameter
        * @param show - DEFAULT FALSE -> if TRUE show projected images
        * @return void
        **/
        static void cylindrical_projection(vector<Mat> input, vector<Mat>& output, double half_fov, bool color = false, bool show = false);


        /**
        * @brief Extract ORB features and compute descriptors for each image
        * @param input       - vector that contains all images
        * @param KeyPoints   - destination matrix to save keypoints, one vector for each image
        * @param Descriptors - destination vector to save descriptors, one Mat for each image
        * @return void
        **/
        static void compute_ORB(vector<Mat> input, vector<vector<KeyPoint>>& KeyPoints, vector<Mat>& Descriptors);


        /**
        * @brief Compute matching features between consecutive images
        * @param Descriptors - vector descriptors, one Mat for each image
        * @param Matches - destination matrix to save matches, one vector for each image
        * @return void
        **/
        static void compute_match(vector<Mat> Descriptors, vector<vector<DMatch>>& Matches);


        /**
        * @brief Refine the matches found selecting the bests
        * @param Matches- vector descriptors, one Mat for each image
        * @param BestMatches - destination matrix to save the best matches, one vector for each image
        * @param ratio - initial parameter used to filter best matches
        * @return void
        **/
        static void filter_best_match(vector<vector<DMatch>> Matches, vector<vector<DMatch>>& BestMatches, float ratio);


        /**
        * @brief Compute translation matrix using RANSAC
        * @param BestMatches - Matrix of the best matches
        * @param KeyPoints   - Matrix of the keypoints
        * @param output      - destination vector where save all the translation matrix, one for couple of adjacent image (should be N-1)
        * @return void
        **/
        static void compute_homography(vector<vector<DMatch>> BestMatches, vector<vector<KeyPoint>> KeyPoints, vector<Mat>& output);


        /**
        * @brief Simple copy&paste merge of images
        * @param img  - Vector of images
        * @param H    - Vector of translation matrices
        * @param save - DEFAULT=FALSE ---> if TRUE save the merged panoramic image inside the path
        * @param path - path where save the merged panoramic image
        * @return void
        **/
        static void merge(vector<Mat> img, vector<Mat> H, bool save=false, const string &path="/");


        /**
        * @brief Edge-blured merge of adjacent images and loop back to the beginning
        * @param img       - Vector of images
        * @param panoramic - Destination of merged panoramic image
        * @param H         - Vector of translation matrices
        * @param delta     - Width of the area in pixels where apply edge-transition effect
        * @param save      - DEFAULT=FALSE ---> if TRUE save the merged panoramic image inside the path
        * @param path      - path where save the merged panoramic image
        * @return void
        **/
        static void blur_edge_merge(vector<Mat> img, Mat& panoramic, vector<Mat> H, int delta, bool save = false, const string& path = "/");



        /**
        * @brief View of merged panoramic image
        * @param img           - Panoramic image
        * @param width         - Input view-window width
        * @param step_size     - Move view left/right step-size
        * @param base_img_cols - Number of columns of a single base-image
        * @return void
        **/
        static void panoramic_viewer(Mat img, int width, int step_size, int base_img_cols);

};
#endif // PANORAMIC_H