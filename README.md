## Active Branches:

1. master
	1. develop (soon to branch out into 157 subtopics. for now, this only contains fourier projects)
		1. fourier-filter
		2. fourier-convolution
		3. fourier-correlation


> Goal: Develop functioning Python scripts (such as animation rendering) for each image processing concept, compile all scripts, and wrap them up into a package (mainly, for visualization purposes but who knows.)



![](https://i.imgur.com/uucRH8l.png)


# Spatial Domain Analysis


To do (Status: 0/6)


> 1. Summary of digital image properties (grayscale, color channels, histograms, CDF) and synthesis
> 2. Summary of color curve manipulation
> 3. Summary of CDF manipulation
> 4. Summary of contrast stretching
> 5. Summary of fade restoration techniques
> 6. Compile into one python script


## Contrast Stretching

```python
contrast_rgb(path, lower, upper)
```


![](https://i.imgur.com/Kl4iMMf.gif)



# Fourier Domain Analysis


To do (Status: 0/1)

> 1. Construct Python script for rendering and fix file structure
> 2. The script should allow the user to:
> 	1. Import an image file
> 	2. Set a *sweeping shape* to vary (the growing circle in the animation)
> 		1. Circle
> 		2. Ring
> 		3. Solid Square
> 		4. Thin Square
> 		5. ! Custom !
> 	3. Generate a sweeping GIF
> 	4. Creates a window with slider to vary size of the sweeping shape
> 		1. Contains ability to save snapshots
> 5. Develop into software and create initial documentation



## Fourier Filtering

```python
render_fourier_filter()
```



![](https://i.imgur.com/mcpxypv.gif)
