�	2:=��5@2:=��5@!2:=��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-2:=��5@�4)�&3@1�x�'e�?A�+e�X�?I�W�ۼ@*	fffff�I@2U
Iterator::Model::ParallelMapV2�(��0�?!�[�þ7@)�(��0�?1�[�þ7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Pk�w�?!X�{��:@)g��j+��?1d�'H=�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2�%䃎?!;�[��<@)M�St$�?1	� U��5@:Preprocessing2F
Iterator::Modelr�����?!\�þA@)��_vOv?1���s�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�R�!�u�?!R�%� uP@)����Mbp?15��U�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!ʼk1��@)��H�}m?1ʼk1��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!�P��@)/n��b?1�P��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ǘ���?!b4�w-F?@)��_�LU?1<A��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��"��X@Qb�?��+�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�4)�&3@�4)�&3@!�4)�&3@      ��!       "	�x�'e�?�x�'e�?!�x�'e�?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	�W�ۼ@�W�ۼ@!�W�ۼ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��"��X@yb�?��+�?�"5
sequential/dense/MatMulMatMul����l�?!����l�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�/���V�?!�uhT��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�~��1�?!n��u!n�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch����l�?!-�.�d�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradNW3�D�?!�h)��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�/���V�?!S���?"7
sequential/dense_1/MatMulMatMul�/���V�?!�A&��?0"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad���ܓ?![��H��?"
sub_1Sub���O��?!99�}��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�G�px��?!�vcw�?Q      Y@Y��/Ċ�0@a�	�N]�T@qo�˦pX@y�j����?"�
both�Your program is POTENTIALLY input-bound because 87.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 