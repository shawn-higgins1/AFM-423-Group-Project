	ӆ���K7@ӆ���K7@!ӆ���K7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ӆ���K7@ԙ{H��4@1dt@���?A�H.�!��?IE���l @*	     �M@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenateg��j+��?![4��C@)��JY�8�?1����cB@:Preprocessing2U
Iterator::Model::ParallelMapV2�?�߾�?!�<�"h87@)�?�߾�?1�<�"h87@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatΈ����?!��c+��/@)�ZӼ�}?1'u_[(@:Preprocessing2F
Iterator::Model�0�*�?!�V'uA@)lxz�,C|?1����c'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa2U0*��?!��}ylEP@)���_vOn?1pR��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?![4��@)/n��b?1[4��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�
F%u�?!��c+��E@)����Mb`?1'u_@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�<�"h�?)��H�}M?1�<�"h�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice-C��6J?!�pR���?)-C��6J?1�pR���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI������X@Q�\�X��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ԙ{H��4@ԙ{H��4@!ԙ{H��4@      ��!       "	dt@���?dt@���?!dt@���?*      ��!       2	�H.�!��?�H.�!��?!�H.�!��?:	E���l @E���l @!E���l @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q������X@y�\�X��?