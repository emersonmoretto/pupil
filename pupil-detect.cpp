/**
 * Program to detect pupil, based on
 * http://www.codeproject.com/Articles/137623/Pupil-or-Eyeball-Detection-and-Extraction-by-C-fro
 * with some improvements.


 LOOKKKK https://github.com/xef6/eyetracker/blob/master/src/cvEyeTracker.cpp @ 1153 !!
 */

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace cv;

int main(int argc, char** argv)
{
	// Load image
	cv::Mat src = cv::imread("Eye_RC.jpg");
	if (src.empty())
		return -1;

	// Invert the source image and convert to grayscale
	cv::Mat gray;
	cv::cvtColor(~src, gray, CV_BGR2GRAY);

	// Convert to binary image by thresholding it
	cv::threshold(gray, gray, 220, 255, cv::THRESH_BINARY);

	// Find all contours
	CvSeq* first_contour = 0;

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// Fill holes in each contour
	cv::drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);


	CvSeq *fin_contour;
	CvSeq *seq_hull;
	CvMemStorage *g_storage[6];

	for(int i=0;i<6;i++){
		g_storage[i] = cvCreateMemStorage(0);
	}

	fin_contour	= cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), g_storage[4]);
	cvClearSeq(fin_contour);

	cvClearMemStorage(g_storage[1]);


	float this_dist;
	float x_avg,y_avg;
	float x_min,x_max,y_min,y_max;
	int point_cnt;
	int tmp_x,tmp_y;
	float max_a = 0;

	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		cv::Rect rect = cv::boundingRect(contours[i]);
		int radius = rect.width/2;

		point_cnt = contours[i].size();
		
		x_avg = 0;
		y_avg = 0;
		for(int j=0;j<point_cnt;j++) {
			tmp_x = contours[i][j].x;
			tmp_y = contours[i][j].y;
			// printf("x,y : %d,%d\n", (int)tmp_x,(int)tmp_y);
			x_avg += tmp_x;
			y_avg += tmp_y;
			if((j==0)||(tmp_x)<(x_min)){ x_min = tmp_x; }
			if((j==0)||(tmp_x)>(x_max)){ x_max = tmp_x; }
			if((j==0)||(tmp_y)<(y_min)){ y_min = tmp_y; }
			if((j==0)||(tmp_y)>(y_max)){ y_max = tmp_y; }
		}
		x_avg /= point_cnt;
		y_avg /= point_cnt;

		// If contour is big enough and has round shape
		// Then it is the pupil
		if (area >= 30 && 
		    std::abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)	
		{
			//cv::circle(src, cv::Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255,0,0), 2);
			for(int k=0;k < point_cnt;k++) {
				cvSeqPush(fin_contour, &contours[i][k]);
			}

		}
	}

	seq_hull = cvConvexHull2( fin_contour, 0, CV_CLOCKWISE, 1);

	CvRect bbox_out = cvRect(0, 0, 0, 0);
	CvBox2D found_ellipse = cvFitEllipse2(seq_hull);

	ellipse(src, cvPoint(found_ellipse.center.x, found_ellipse.center.y),
					  cvSize(found_ellipse.size.width*0.5, found_ellipse.size.height*0.5),
					  found_ellipse.angle, 0, 360.0, CV_RGB(255,0,0),2,8,0);


	if(seq_hull != NULL){
		if(seq_hull->total >= 6){ // only 6 required
			CvRect pupil_bounds = cvBoundingRect(seq_hull,1);
			bbox_out.x = pupil_bounds.x;
			bbox_out.y = pupil_bounds.y;
			bbox_out.width = pupil_bounds.width;
			bbox_out.height = pupil_bounds.height;

			CvBox2D this_fit = cvFitEllipse2(seq_hull);

			float eccentricity = this_fit.size.width/this_fit.size.height;
			if(eccentricity>1.0) eccentricity = 1.0 / eccentricity;

			if(eccentricity>0.35){
				found_ellipse.size   = cvSize2D32f(MIN(MAX(this_fit.size.width,1),999),
													MIN(MAX(this_fit.size.height,1),999));
				found_ellipse.center = cvPoint2D32f(MIN(MAX(this_fit.center.x,0),999),
													 MIN(MAX(this_fit.center.y,0),999));
				found_ellipse.angle  = this_fit.angle;

				printf("x,y,angle : %d,%d,%f\n", (int)found_ellipse.center.x,(int)found_ellipse.center.y, found_ellipse.angle);

			}

		}
	}


	ellipse(src, cvPoint(found_ellipse.center.x, found_ellipse.center.y),
					  cvSize(found_ellipse.size.width*0.5, found_ellipse.size.height*0.5),
					  found_ellipse.angle, 0, 360.0, CV_RGB(255,255,255),1,8,0);



	cv::imshow("image", src);
	cv::waitKey(0);

	return 0;
}