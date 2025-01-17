�	��-�|I7@��-�|I7@!��-�|I7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��-�|I7@��_Yiv4@1f���i�?Aa��+e�?I/�o�ӥ@*	gfffffE@2U
Iterator::Model::ParallelMapV2�0�*�?!���f�8@)�0�*�?1���f�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!�N]���;@)U���N@�?1x�Zn�5@:Preprocessing2F
Iterator::Model�o_��?!bEi|d�C@)-C��6z?1*J�#�-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��~j�t�?!��4>26@)�����w?16��XQ+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!�f�+J!@)���_vOn?1�f�+J!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}гY���?!�����}N@)y�&1�l?1�L�w�Z @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?!�	�N]@){�G�zd?1�	�N]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!      9@)a2U0*�S?1�w�Zn@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���W�X@Q�;]��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��_Yiv4@��_Yiv4@!��_Yiv4@      ��!       "	f���i�?f���i�?!f���i�?*      ��!       2	a��+e�?a��+e�?!a��+e�?:	/�o�ӥ@/�o�ӥ@!/�o�ӥ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���W�X@y�;]��?�"5
sequential/dense/MatMulMatMul����dң?!����dң?0"C
%gradient_tape/sequential/dense/MatMulMatMulr�y����?!�M!�{��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulr�y����?!�&^Ň�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��PX�?!S����?0"7
sequential/dense_1/MatMulMatMul�Qu�yF�?!�Z�k�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulA7�?!�^3�M,�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamr�y����?! ��G `�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchr�y����?!N�њ��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tiler�y����?!�!����?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradr�y����?!������?Q      Y@Y>����/@aX�i��U@qg�Bf��X@y�&^Ň�?"�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 