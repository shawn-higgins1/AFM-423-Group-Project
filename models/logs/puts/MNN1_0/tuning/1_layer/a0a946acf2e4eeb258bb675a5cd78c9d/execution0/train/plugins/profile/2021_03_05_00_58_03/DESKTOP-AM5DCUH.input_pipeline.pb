	���k5@���k5@!���k5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���k5@��l 53@1� "5�b�?A)�{�i¦?IGXT��d�?*	433333G@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�&1��?!5�rO#,>@)�~j�t��?1	�=���9@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!sO#,��7@)�I+��?1sO#,��7@:Preprocessing2F
Iterator::Model�:pΈ�?!��{�C@)�ZӼ�}?1���{�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ׁsF�?!a��V5@)a��+ey?1�{a�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!GX�i��@)���_vOn?1GX�i��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!X�i��~N@)�~j�t�h?1	�=���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!����=@)����Mb`?1����=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�St$�?!���{Z8@)Ǻ���V?1�4�rO#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��]ch�X@QO�
Q���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��l 53@��l 53@!��l 53@      ��!       "	� "5�b�?� "5�b�?!� "5�b�?*      ��!       2	)�{�i¦?)�{�i¦?!)�{�i¦?:	GXT��d�?GXT��d�?!GXT��d�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��]ch�X@yO�
Q���?