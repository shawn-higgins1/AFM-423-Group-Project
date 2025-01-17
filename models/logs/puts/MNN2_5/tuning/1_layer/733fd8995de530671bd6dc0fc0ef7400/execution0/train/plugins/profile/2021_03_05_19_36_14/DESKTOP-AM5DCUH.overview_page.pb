�	K�R��!5@K�R��!5@!K�R��!5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-K�R��!5@,��yp�2@1F~�,�?A0*��D�?I��!�Z�?*	ffffffM@2U
Iterator::Model::ParallelMapV2�� �rh�?!�7�L\�<@)�� �rh�?1�7�L\�<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�o_��?!�Q���f<@)a��+e�?1F�_��5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!��+�6@)�g��s��?1e�J�2@:Preprocessing2F
Iterator::Model��@��ǘ?!e�J��D@)��H�}}?1�Cc}(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"��u��q?!P&�to@@)"��u��q?1P&�to@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe�X��?!���kM@)y�&1�l?1�<��<�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!����S@)a2U0*�c?1����S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�:pΈ�?!���E�>@)Ǻ���V?1�0�0@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIe���X@Q�&��|�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,��yp�2@,��yp�2@!,��yp�2@      ��!       "	F~�,�?F~�,�?!F~�,�?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	��!�Z�?��!�Z�?!��!�Z�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qe���X@y�&��|�?�"5
sequential/dense/MatMulMatMulY��ƔǦ?!Y��ƔǦ?0"C
%gradient_tape/sequential/dense/MatMulMatMul3����?�?!F������?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchɠ܈O�?!�ݰ�|׻?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulɠ܈O�?!�씯�?"7
sequential/dense_1/MatMulMatMulɠ܈O�?!3����?�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��}֘?!�Y�bZ�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam3����?�?!n�)V��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile���-��?!�AXo*`�?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad皛i$u�?!C������?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam���q_�?!0�cZ�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�ܨpaX@y��N���?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 