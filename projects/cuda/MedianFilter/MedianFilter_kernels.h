/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __MEDIANFILTER_KERNELS_H_
#define __MEDIANFILTER_KERNELS_H_

typedef unsigned char Pixel;

// global determines which filter to invoke
enum MedianDisplayMode
{
    MEDIANDISPLAY_IMAGE = 0,
    MEDIANDISPLAY_SOBELTEX,
    MEDIANDISPLAY_SOBELSHARED
};


extern enum MedianDisplayMode g_MedianDisplayMode;

extern "C" void medianFilter(Pixel *odata, int iw, int ih, enum MedianDisplayMode mode, float fScale);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);

#endif

