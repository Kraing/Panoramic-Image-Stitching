#include "panoramic.h"


int main()
{   
    // Set intial parameters
    // dolomites -> #img = 23 / HFOV = 27
    // lab       -> #img = 12 / HFOV = 33
    // kitchen   -> #img = 20 / HFOV = 33
    int image_number = 23;
    double half_fov = 27.;
    string img_path = "dolomites/";

    // Load source images
    vector<Mat> src = Panoramic::load_images(img_path, image_number);


    // Project source img in to cylindrical view
    vector<Mat> cylindrical_src;
    Panoramic::cylindrical_projection(src, cylindrical_src, half_fov, true);


    // ------ Compute keypoints and descriptors for each image -------------------------------
    vector<vector<KeyPoint>> kp;
    vector<Mat> descriptors;
    Panoramic::compute_ORB(cylindrical_src, kp, descriptors);
    

    // ------ Compute match vectors ----------------------------------------------------------
    vector<vector<DMatch>> match;
    Panoramic::compute_match(descriptors, match);


    // ------- Filter the best matches for each couple of adjacent images --------------------
    float ratio = 1.5;
    vector<vector<DMatch>> BestMatches;
    Panoramic::filter_best_match(match, BestMatches, ratio);


    // ----- Compute translation matrix H for each couple of adjacent images -----------------
    vector<Mat> H;
    Panoramic::compute_homography(BestMatches, kp, H);


    // ---------------- Copy&Paste merge of the images ---------------------------------------
    Panoramic::merge(cylindrical_src, H, true, img_path);


    // ---------------- Edge-transition merge of the images ----------------------------------
    Mat pano;
    int delta = 40;
    Panoramic::blur_edge_merge(cylindrical_src, pano, H, delta, true, img_path);


    // --------------- Panoramic Viewer Visualization -----------------------------------------
    int width = 1000;
    int step_size = 50;
    int base_img_col = cylindrical_src[0].cols;
    Panoramic::panoramic_viewer(pano, width, step_size, base_img_col);
}