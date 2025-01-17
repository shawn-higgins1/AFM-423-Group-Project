�	s��{�5@s��{�5@!s��{�5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-s��{�5@�V|C�73@1Ͽ]��N�?A�3��7�?I���0�z @*	������F@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate"��u���?!�}��B@)X9��v��?1�#����@@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ����?!��#��8@)Ǻ����?1��#��8@:Preprocessing2F
Iterator::Model�5�;Nё?!Dy�5C@)a��+ey?1�YLg1+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!�}�,@)U���N@s?1�YLg1�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�!��u��?!�����N@)F%u�k?1)�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!1��t�@)_�Q�[?11��t�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�l����?!��>��HD@)��_�LU?1�,����@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!���b:�@)����MbP?1���b:�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��#���?)Ǻ���F?1��#���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�H\��X@Q�'��Q��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�V|C�73@�V|C�73@!�V|C�73@      ��!       "	Ͽ]��N�?Ͽ]��N�?!Ͽ]��N�?*      ��!       2	�3��7�?�3��7�?!�3��7�?:	���0�z @���0�z @!���0�z @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�H\��X@y�'��Q��?�"5
sequential/dense/MatMulMatMul�O���?!�O���?0"C
%gradient_tape/sequential/dense/MatMulMatMulh�q�ݨ�?!��f��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�O���?!�����B�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�O���?!;�c��3�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�O���?!3h��$F�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamh�q�ݨ�?!`�T@��?"!
Adam/PowPowh�q�ݨ�?!��
\0�?"E
'gradient_tape/sequential/dense_1/MatMulMatMulh�q�ݨ�?!� �w��?0"7
sequential/dense_1/MatMulMatMulh�q�ݨ�?!�T.v��?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileW�`���?!,bt����?Q      Y@Y{	�%��1@a�����T@q�0Y��X@y�j���	�?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 