	-|}�K�3@-|}�K�3@!-|}�K�3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails--|}�K�3@���FX2@1�U�@�?A�~j�t��?I@�&M�"�?*������H@)       =2U
Iterator::Model::ParallelMapV2%u��?!��&�l�=@)%u��?1��&�l�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!E]t�E;@)������?1ż�!1o7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!m��&�l6@)� �	�?1�q�q/@:Preprocessing2F
Iterator::Modelj�t��?!c�ΐ��E@)_�Q�{?1$���y+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!�!1ogHL@)_�Q�k?1$���y@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice_�Q�k?!$���y@)_�Q�k?1$���y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!
����@)ŏ1w-!_?1
����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�nu�X@QB�<}���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���FX2@���FX2@!���FX2@      ��!       "	�U�@�?�U�@�?!�U�@�?*      ��!       2	�~j�t��?�~j�t��?!�~j�t��?:	@�&M�"�?@�&M�"�?!@�&M�"�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�nu�X@yB�<}���?