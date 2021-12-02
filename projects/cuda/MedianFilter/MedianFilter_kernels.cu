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

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_string.h>

#include "MedianFilter_kernels.h"

// Texture object for reading image
cudaTextureObject_t texObject;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__device__ unsigned char ComputeMedianFilter(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short temp[9] = { ul, um,ur,ml,mm,mr,ll,lm,lr };
    for (int i = 0; i < 9 - 1; i++)
    {
        for (int j = 0; j < 9 - i; j++)
        {
            if (temp[j] > temp[j + 1])
            {
                int t = temp[j];
                temp[j] = temp[j + 1];
                temp[j + 1] = t;
            }
        }
    }

    return (unsigned char)temp[4];
}


__global__ void
SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch,
               int w, int h, float fscale, cudaTextureObject_t tex)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        pSobel[i] = min(max((tex2D<unsigned char>(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
    }
}

__global__ void
MedianTex(Pixel *pSobelOriginal, unsigned int Pitch,
         int w, int h, float fScale, cudaTextureObject_t tex)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        unsigned char pix00 = tex2D<unsigned char>(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D<unsigned char>(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D<unsigned char>(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D<unsigned char>(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D<unsigned char>(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D<unsigned char>(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D<unsigned char>(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D<unsigned char>(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D<unsigned char>(tex, (float) i+1, (float) blockIdx.x+1);
        pSobel[i] = ComputeMedianFilter(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
    }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;

    if (Bpp == 1)
    {
        desc = cudaCreateChannelDesc<unsigned char>();
    }
    else
    {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, iw * Bpp * sizeof(Pixel), 
                                        iw * Bpp * sizeof(Pixel), ih, cudaMemcpyHostToDevice));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = array;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

}

extern "C" void deleteTexture(void)
{
    checkCudaErrors(cudaFreeArray(array));
    checkCudaErrors(cudaDestroyTextureObject(texObject));
}


// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void medianFilter(Pixel *odata, int iw, int ih, enum MedianDisplayMode mode, float fScale)
{
    switch (mode)
    {
        case MEDIANDISPLAY_IMAGE:
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale, texObject);
            break;

        case MEDIANDISPLAY_SOBELTEX:
            MedianTex <<<ih, 384>>>(odata, iw, iw, ih, fScale, texObject);
            break;
    }
}
