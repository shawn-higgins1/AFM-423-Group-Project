�	
�g��2@
�g��2@!
�g��2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-
�g��2@���v��1@1��
���?AG�tF^�?Ih���x�?*	    @K@2U
Iterator::Model::ParallelMapV2vq�-�?!�O����<@)vq�-�?1�O����<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���{�?!�߅��=@)a��+e�?1
l�O��6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!��[��6@)��ZӼ�?1s�3R1�2@:Preprocessing2F
Iterator::Model�z6�>�?!6�'K`�D@)lxz�,C|?1$s�3R)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�q����o?!��ش�@)�q����o?1��ش�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���H�?!�ش�,M@)-C��6j?1�w� z|@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!6�'K`�@)�J�4a?16�'K`�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIN4��X@Q���?��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���v��1@���v��1@!���v��1@      ��!       "	��
���?��
���?!��
���?*      ��!       2	G�tF^�?G�tF^�?!G�tF^�?:	h���x�?h���x�?!h���x�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qN4��X@y���?��?�"5
sequential/dense/MatMulMatMul��|�>ƫ?!��|�>ƫ?0"C
%gradient_tape/sequential/dense/MatMulMatMulx�S�}c�?!�h�Sޔ�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�üEܞ?!0������?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�üEܞ?!%����?"7
sequential/dense_1/MatMulMatMul�üEܞ?!�P	]�?0"!
Adam/PowPows6�7��?!BOJs�?"S
2gradient_tape/sequential/dense/BiasAdd/BiasAddGradBiasAddGrad81%۴��?!i�h����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam֍��)��?!�/&��?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam֍��)��?!o���h��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam֍��)��?!L$K���?Q      Y@Y�M�_{4@a��(�S@q��kW�W@yj>���I�?"�
both�Your program is POTENTIALLY input-bound because 92.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 