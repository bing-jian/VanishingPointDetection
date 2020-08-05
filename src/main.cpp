extern "C"
{
    #include "lsd.h"
};
#include "VPDetection.h"

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;


// LSD line segment detection
void LineDetect( cv::Mat image, double thLength, std::vector<std::vector<double> > &lines )
{
	cv::Mat grayImage;
	if ( image.channels() == 1 )
		grayImage = image;
	else
		cv::cvtColor(image, grayImage, CV_BGR2GRAY);

	image_double imageLSD = new_image_double( grayImage.cols, grayImage.rows );
	unsigned char* im_src = (unsigned char*) grayImage.data;

	int xsize = grayImage.cols;
	int ysize = grayImage.rows;
	for ( int y = 0; y < ysize; ++y )
	{
		for ( int x = 0; x < xsize; ++x )
		{
			imageLSD->data[y * xsize + x] = im_src[y * xsize + x];
		}
	}

	ntuple_list linesLSD = lsd( imageLSD );
	free_image_double( imageLSD );

	int nLines = linesLSD->size;
	int dim = linesLSD->dim;
	std::vector<double> lineTemp( 4 );
	for ( int i = 0; i < nLines; ++i )
	{
		double x1 = linesLSD->values[i * dim + 0];
		double y1 = linesLSD->values[i * dim + 1];
		double x2 = linesLSD->values[i * dim + 2];
		double y2 = linesLSD->values[i * dim + 3];

		double l = sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) );
		if ( l > thLength )
		{
			lineTemp[0] = x1;
			lineTemp[1] = y1;
			lineTemp[2] = x2;
			lineTemp[3] = y2;

			lines.push_back( lineTemp );
		}
	}

	free_ntuple_list(linesLSD);
}

void drawClusters( cv::Mat &img, std::vector<std::vector<double> > &lines, std::vector<std::vector<int> > &clusters )
{
	int cols = img.cols;
	int rows = img.rows;

	//draw lines
	std::vector<cv::Scalar> lineColors( 3 );
	lineColors[0] = cv::Scalar( 0, 0, 255 );  // red
	lineColors[1] = cv::Scalar( 0, 255, 0 );  // green
	lineColors[2] = cv::Scalar( 255, 0, 0 );  // blue

	for ( int i=0; i<lines.size(); ++i )
	{
		int idx = i;
		cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1]);
		cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3]);
		cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

		cv::line( img, pt_s, pt_e, cv::Scalar(0,0,0), 2, CV_AA );
	}

	for ( int i = 0; i < clusters.size(); ++i )
	{
		for ( int j = 0; j < clusters[i].size(); ++j )
		{
			int idx = clusters[i][j];

			cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
			cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );
			cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

			cv::line( img, pt_s, pt_e, lineColors[i], 2, CV_AA );
		}
	}
}

int main(int argc, char* argv[])
{
	string inPutImage(argv[1]);

	cv::Mat image= cv::imread( inPutImage );
	if ( image.empty() )
	{
		printf( "Load image error : %s\n", inPutImage.c_str() );
	}

	// LSD line segment detection
	double thLength = 50.0;
	std::vector<std::vector<double> > lines;
	LineDetect( image, thLength, lines );

	// Camera internal parameters
    cv::Size size = image.size();
	cv::Point2d pp(size.width * 0.5, size.height * 0.5);        // Principle point (in pixel)
	double f = std::max(size.width, size.height) * 1.2;          // Focal length (in pixel)


	// Vanishing point detection
	std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
	std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
	VPDetection detector;
	detector.run( lines, pp, f, vps, clusters );

	drawClusters( image, lines, clusters );
	imshow("",image);
	cv::waitKey( 0 );
}
