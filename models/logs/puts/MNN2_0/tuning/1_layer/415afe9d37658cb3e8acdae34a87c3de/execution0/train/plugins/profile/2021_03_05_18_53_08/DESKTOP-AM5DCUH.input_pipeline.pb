	��cg5@��cg5@!��cg5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��cg5@�q�
�2@1���DR�?A��0�*�?Iǃ-v�,@*�����YJ@)       =2U
Iterator::Model::ParallelMapV2����Mb�?!\���K\>@)����Mb�?1\���K\>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�&S��?!D]�L~DA@)����Mb�?1\���K\>@:Preprocessing2F
Iterator::Model
ףp=
�?!Y'aH�XE@)9��v��z?1�`7���(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	�^)ˀ?!/qy�/@) �o_�y?1��l��'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[���?!�؞��L@)Ǻ���f?1A�'r�@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�|d��@)ŏ1w-!_?1�|d��@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��_�LU?!�����@)��_�LU?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*��?!7���`7B@)����MbP?1\���K\�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��H�}M?!S[| S�?)��H�}M?1S[| S�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��_�-�X@Q8!P*��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�q�
�2@�q�
�2@!�q�
�2@      ��!       "	���DR�?���DR�?!���DR�?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	ǃ-v�,@ǃ-v�,@!ǃ-v�,@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��_�-�X@y8!P*��?