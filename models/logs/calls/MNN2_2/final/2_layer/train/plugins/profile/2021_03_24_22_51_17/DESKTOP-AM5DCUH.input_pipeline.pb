	����Z7@����Z7@!����Z7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����Z7@��Ͻ4@1��rg&�?A��n�;2�?I���D�K@*	������I@2U
Iterator::Model::ParallelMapV2?W[���?!     ~=@)?W[���?1     ~=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate���S㥋?!     ^:@)n���?1     $3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;�O��?!     v6@)�j+��݃?1     �2@:Preprocessing2F
Iterator::Modelj�t��?!     �D@)-C��6z?1      )@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice���_vOn?!     �@)���_vOn?1     �@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?!     M@)F%u�k?1     �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!      @)��H�}]?1      @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2�%䃎?!     =@)Ǻ���V?1     �@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�%��X@Q�v�~^�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Ͻ4@��Ͻ4@!��Ͻ4@      ��!       "	��rg&�?��rg&�?!��rg&�?*      ��!       2	��n�;2�?��n�;2�?!��n�;2�?:	���D�K@���D�K@!���D�K@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�%��X@y�v�~^�?