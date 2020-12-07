#include "panoramic.h"
#include "panoramic_utils.h"


// Support function for filter_best_match function
vector<DMatch> autotune_matches(vector<DMatch> Matches, float min_dist, float ratio)
{
    vector<DMatch> match;
    for (int j = 0; j < int(Matches.size()); j++)
    {
        if (Matches[j].distance < min_dist * ratio)
            match.push_back(Matches[j]);
    }
    return match;
}


// Load images from given path and number
vector<Mat> Panoramic::load_images(string path, int n, bool show)
{
    // Initialize variables
    vector<Mat> src;
    string tmp;

    string type1 = ".bmp";
    string type2 = ".png";
    string type3 = ".jpg";
    string utype;

    size_t found = path.find("dolomites");

    if (found != string::npos)
        utype = type2;
    else
    {
        // test purpose
        found = path.find("mi8");
        if (found != string::npos)
            utype = type3;
        else
            utype = type1;
    }


    // loop for each image
    for (int i = 0; i < n; i++)
    {  
        if (i < 9)
            tmp = path + "i0" + to_string(i + 1) + utype;
        else
            tmp = path + "i" + to_string(i + 1) + utype;

        // temporary variable to store and test loaded image
        Mat img = imread(tmp, IMREAD_COLOR);

        if (img.empty())
            cerr << "Errors " << tmp << " can't be loaded." << endl;
        else
        {
            cout << "Image" << tmp << "successfully loaded." << endl;

            // push the image inside MAT-vector
            src.push_back(img);
            // Show the image if needed
            if (show)
            {
                namedWindow(tmp);
                imshow(tmp, src[i]);
                waitKey(0);
            }
        }
    }
    return src;
}


// Project given images to cylindrical surfaceand converto to grey-scale or color
void Panoramic::cylindrical_projection(vector<Mat> input, vector<Mat>& output, double half_fov, bool color, bool show)
{
    // Loop over each image to create the projection
    for (int i = 0; i < input.size(); i++)
    {
        if (color)
            output.push_back(PanoramicUtils::cylindricalProj_color(input[i], half_fov));
        else
            output.push_back(PanoramicUtils::cylindricalProj(input[i], half_fov));

        cout << "Image-" << i << "successfully projected." << endl;
        // Show the image if needed
        if (show)
        {
            string tmp = "im" + to_string(i);
            namedWindow(tmp);
            imshow(tmp, output[i]);
            waitKey(0);
        }
    }
    return;
}


// Extract ORB features and compute descriptors for each image
void Panoramic::compute_ORB(vector<Mat> input, vector<vector<KeyPoint>>& KeyPoints, vector<Mat>& Descriptors)
{
    // Create default ORB Object
    Ptr<Feature2D> orb = ORB::create();

    for (int i = 0; i < input.size(); i++)
    {
        // Initialize temporrary variable to be pushed insisde the input keypoints and descriptors vector
        Mat tmp_Mat;
        vector<KeyPoint> tmp_kp;

        // Compute keypoints and descriptor of i-th image
        orb->detectAndCompute(input[i], Mat(), tmp_kp, tmp_Mat);

        Descriptors.push_back(tmp_Mat);
        KeyPoints.push_back(tmp_kp);

        //cout << Descriptors[i] << endl;
        //cout << "Feature and descriptor of image " << i + 1 << " computed." << endl;
    }
    return;
}


// Compute matching features between consecutive images
void Panoramic::compute_match(vector<Mat> Descriptors, vector<vector<DMatch>>& Matches)
{

    vector<DMatch> tmp_matches;

    // 4 should be the norm-type ENUM that correspond to NORM_HAMMING --- we use also cross-match
    Ptr<BFMatcher> matcher = BFMatcher::create(4, true);

    for (int i = 0; i < Descriptors.size() - 1; i++)
    {
        //cout << "Size of " << i << "-th descriptor: " << Descriptors[i].size() << endl;
        
        matcher->match(Descriptors[i], Descriptors[i + 1], tmp_matches, Mat());

        Matches.push_back(tmp_matches);
        cout << "Match between " << i + 1 << "-" << i + 2 << " computed." << endl;
    }

    // Add last-first match
    matcher->match(Descriptors.back(), Descriptors[0], tmp_matches, Mat());
    Matches.push_back(tmp_matches);
    cout << "Match between last-first computed." << endl;
    return;
}


// Filter the best matches (default select at least 120)
void Panoramic::filter_best_match(vector<vector<DMatch>> Matches, vector<vector<DMatch>>& BestMatches, float ratio)
{
    for (int i = 0; i < Matches.size(); i++)
    {
        float instance_ratio = ratio;

        // push in bestMatches the top 50 matched features
        vector<DMatch> bestMatches;
        int nbMatch = int(Matches[i].size());
        Mat tab(nbMatch, 1, CV_32F);

        float dist;
        float min_dist = -1.;

        // Find the minumun distance between matchpoints
        for (int j = 0; j < nbMatch; j++)
        {
            dist = Matches[i][j].distance;

            // update the minumun distance
            if (min_dist < 0 || dist < min_dist)
                min_dist = dist;
        }
        

        // Adapt the ratio in order to get at least 120 matches per couple of adjacent images
        do
        {
            bestMatches = autotune_matches(Matches[i], min_dist, instance_ratio);
            instance_ratio = 2 * instance_ratio;
        } while (bestMatches.size() < 120);
        

        BestMatches.push_back(bestMatches);
        //cout << "Size of " << i << "-th" << "TOP matches: " << BestMatches[i].size() << endl;

    }

    return;
}


// Compute translation matrix using RANSAC
void Panoramic::compute_homography(vector<vector<DMatch>> BestMatches, vector<vector<KeyPoint>> KeyPoints, vector<Mat>& output)
{                

    cout << "Compute Homography: " << endl;

    // Loop over each image
    for (int i = 0; i < BestMatches.size() - 1; i++)
    {
        // Initialize variables for storing points
        vector<Point2f> src_pts;
        vector<Point2f> dst_pts;

        // Loop inside i-th image
        for (vector<DMatch>::iterator it = BestMatches[i].begin(); it != BestMatches[i].end() - 1; ++it)
        {
            //-- Get the keypoints from the good matches
            src_pts.push_back(KeyPoints[i + 1][it->trainIdx].pt);
            dst_pts.push_back(KeyPoints[i][it->queryIdx].pt);
        }

        // Compute homography
        Mat H = findHomography(src_pts, dst_pts, RANSAC);

        /* Force homography to NOT rotate the image keeping only x-coordinate
        // translation for test purpose
        H.at<double>(Point(0, 0)) = 1.;
        H.at<double>(Point(1, 0)) = 0.;
        H.at<double>(Point(0, 1)) = 0.;
        H.at<double>(Point(1, 1)) = 1.;
        H.at<double>(Point(0, 2)) = 0.;
        H.at<double>(Point(1, 2)) = 0.;
        */

        // Append the homography to output vector
        output.push_back(H);
        cout << "H-" << i << "-" << i + 1 << endl;
        waitKey(0);
    }

    // find the last-first homography and push inside H
    vector<Point2f> src_pts;
    vector<Point2f> dst_pts;
    for (vector<DMatch>::iterator it = BestMatches.back().begin(); it != BestMatches.back().end() - 1; ++it)
    {
        src_pts.push_back(KeyPoints[0][it->trainIdx].pt);
        dst_pts.push_back(KeyPoints.back()[it->queryIdx].pt);
    }
    Mat H = findHomography(src_pts, dst_pts, RANSAC);
    output.push_back(H);
    cout << "H-LAST-FIRST" << endl << endl;

    return;
}


// Simple copy&paste merge of images
void Panoramic::merge(vector<Mat> img, vector<Mat> H, bool save, const string& path)
{   
    // Initialize x-offset vector and add the first offset to 0
    vector<double> cumsum_x;
    double last_first_offset;
    cumsum_x.push_back(0.);

    // cumsum until last -> last first is not inside
    for (int i = 0; i < H.size(); i++)
    {
        double tmp = H[i].at<double>(Point(2, 0));

        // If is the first H just push the offset
        if (i == 0)
            cumsum_x.push_back(tmp);
        else
        {   
            // Else cumulative sum the new offset with the previous one
            cumsum_x.push_back(tmp + cumsum_x[i]);
        }
    }

    // Initialize the canvas
    Mat panoramic_canvas(Size((int)cumsum_x.back() + img[0].cols, img[0].rows), img[0].type());
    panoramic_canvas = Scalar(0);

    for (int i = 0; i < img.size(); i++)
    {
        // Copy and paste translated images inside the panoramic-canvas
        img[i].copyTo(panoramic_canvas(Rect(cumsum_x[i], 0, img[i].cols, img[i].rows)));
    }

    if(save)
        imwrite(path + "CP_Panoramic.jpg", panoramic_canvas);

    return;
}


// Edge-blured merge of adjacent images and loop back to the beginning
void Panoramic::blur_edge_merge(vector<Mat> img, Mat& panoramic, vector<Mat> H, int delta, bool save, const string& path)
{
    // Initialize x-offset vector and add the first offset to 0
    vector<int> cumsum_x;
    cumsum_x.push_back(0.);

    for (int i = 0; i < H.size(); i++)
    {
        double tmp = H[i].at<double>(Point(2, 0));

        // If is the first H just push the offset
        if (i == 0)
            cumsum_x.push_back(tmp);
        else
        {
            // Else cumulative sum the new offset with the previous one
            cumsum_x.push_back(tmp + cumsum_x[i]);
        }
    }

    // Initialize the panoramic canvas
    Mat panoramic_canvas(Size((int)cumsum_x.back() + img[0].cols - 1, img[0].rows), img[0].type());
    panoramic_canvas = Scalar(0);

    // Define blur transition step-size
    double delta_alpha = 1. / delta;
    int last_offset = (int)H.back().at<double>(Point(2, 0));

    for (int i = 0; i < img.size() - 1; i++)
        {
            // If first image -> copy the initial part directly
            if (i == 0)
            {
                for (int j = 0; j < img[i].cols; j++)
                {
                    img[i].col(j).copyTo(panoramic_canvas.col(j));
                }
            }

            // copy the second part of the image as row -> will be rewritten in the next iteration
            for (int k = 0; k < (int)H[i].at<double>(Point(2, 0)); k++)
            {
                img[i + 1].col(img[i + 1].cols - (int)H[i].at<double>(Point(2, 0)) + k).copyTo(panoramic_canvas.col(img[i + 1].cols + cumsum_x[i] + k));
            }

            // Compute the overlap blur before the end of the first image
            int counter = 1;
            for (int j = img[i].cols + (int)floor(cumsum_x[i]) - delta; j < img[i].cols + (int)floor(cumsum_x[i]); j++)
            {

                Mat tmp_col;
                addWeighted(panoramic_canvas.col(j), (1 - counter * delta_alpha), img[i + 1].col(j - (int)floor(cumsum_x[i]) - (int)H[i].at<double>(Point(2, 0))), (counter * delta_alpha), 0., tmp_col);
                counter++;
                tmp_col.copyTo(panoramic_canvas.col(j));
            }
        }

    // add the last-first merge
    // Compute the overlap blur between last-first images
    int cumsum_size = cumsum_x.size();
    int counter = 1;
    for (int j = img.back().cols + cumsum_x[cumsum_size - 2] - delta; j < img.back().cols + (int)(cumsum_x[cumsum_size - 2] - 1); j++)
        {

            Mat tmp_col;
            addWeighted(panoramic_canvas.col(j), (1 - counter * delta_alpha), panoramic_canvas.col(j - cumsum_x[cumsum_size - 2] - (int)H.back().at<double>(Point(2, 0))), (counter * delta_alpha), 0., tmp_col);
            counter++;
            // copy the blurred transition to the edge between last-first image
            tmp_col.copyTo(panoramic_canvas.col(j));
            cout << "blur-idx: " << j << endl;
        }

    // copy the second part of the image as row
    int tmp_len = H.size();
    for (int k = 0; k < (int)H.back().at<double>(Point(2, 0)); k++)
        {
            int temp = (int)H[tmp_len - 1].at<double>(Point(2, 0));
            panoramic_canvas.col(img[0].cols - temp + k).copyTo(panoramic_canvas.col(img[0].cols + cumsum_x[tmp_len - 1] + k - 1));
            cout << "Last-idx: " << img[0].cols + cumsum_x[tmp_len - 1] + k << endl;
        }

    // copy the attachment of last first inside the first image
    for (int n = 0; n < img[0].cols - 1; n++)
        {
            panoramic_canvas.col((int)cumsum_x.back() + n).copyTo(panoramic_canvas.col(n));
        }
        
        
    // Save the image adding the used parameter value
    if (save)
        imwrite(path + "EB_Panoramic_D" + to_string(delta) + ".jpg", panoramic_canvas);
    

    panoramic = panoramic_canvas;
    return;
}


// View of merged panoramic image
void Panoramic::panoramic_viewer(Mat img, int width, int step_size, int base_img_cols)
{
    // test arrow key output
    int max_width = img.cols;
    int scroll_idx = 0;

    // Create canvas
    Mat canvas(Size(width, img.rows), img.type());
    canvas = Scalar(0);

    imshow("Panoramic_Viewer", img(Rect(0, 0, width, img.rows)));
    do
    {
        int key = waitKey(0);
        cout << "Pressed Key: " << key << endl;

        // break if `esc' key was pressed. 
        if (key == 27) return;
        
        // if "d" pressed rotate left
        if (key == 100)
        {
            // Manage the case when we are in the merge area rotating left
            if (scroll_idx < 0)
                scroll_idx = max_width + scroll_idx - base_img_cols;

            // Update scroll-index
            scroll_idx += step_size;
            cout << "idx: " << scroll_idx << endl;

            // Reset scroll-index if returned at the beginning
            if (scroll_idx >= max_width)
                scroll_idx = scroll_idx - max_width + base_img_cols;

            // Show last-first merge connection
            if (scroll_idx > max_width - width)
            {
                //scroll_idx -= step_size;
                int tmp = scroll_idx + width - max_width;

                // copy last part of the end
                img(Rect(0 + scroll_idx, 0, width - tmp, img.rows)).copyTo(canvas.colRange(0, width - tmp).rowRange(0, img.rows));

                // copyt the initial part of the beginning
                img(Rect(0 + base_img_cols, 0, tmp, img.rows)).copyTo(canvas.colRange(width - tmp, width).rowRange(0, img.rows));
                
                imshow("Panoramic_Viewer", canvas); 
                continue;
            }
            
            // Show normal case - copy view inside canvas
            img(Rect(0 + scroll_idx, 0, width, img.rows)).copyTo(canvas.colRange(0, width).rowRange(0, img.rows));
            imshow("Panoramic_Viewer", canvas);
        }

        // if "a" pressed rotate right
        if (key == 97)
        {
            // Manage the case when we are in the merge area rotating right
            if (scroll_idx > max_width - width)
                scroll_idx = scroll_idx - max_width + base_img_cols;

            scroll_idx -= step_size;
            cout << "idx: " << scroll_idx << endl;

            // Reset the index
            if (scroll_idx <= -width)
                scroll_idx = max_width + scroll_idx - base_img_cols;

            if (scroll_idx < 0)
            {
                //scroll_idx = max_width + scroll_idx;
                int tmp_idx = scroll_idx - width + max_width;
                int tmp = tmp_idx + width - max_width;

                // copy last part of the end
                img(Rect(max_width - base_img_cols + scroll_idx, 0, - scroll_idx, img.rows)).copyTo(canvas.colRange(0, - scroll_idx).rowRange(0, img.rows));

                // copyt the initial part of the beginning
                img(Rect(0, 0, width + scroll_idx, img.rows)).copyTo(canvas.colRange(- scroll_idx, width).rowRange(0, img.rows));

                imshow("Panoramic_Viewer", canvas);
                continue;
            }

            img(Rect(0 + scroll_idx, 0, width, img.rows)).copyTo(canvas.colRange(0, width).rowRange(0, img.rows));
            imshow("Panoramic_Viewer", img(Rect(0 + scroll_idx, 0, width, img.rows)));
        }
    } while (true);
}