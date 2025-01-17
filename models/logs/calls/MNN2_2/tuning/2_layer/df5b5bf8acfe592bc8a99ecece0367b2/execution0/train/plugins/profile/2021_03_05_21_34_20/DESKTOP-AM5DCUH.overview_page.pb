�	
��$>G8@
��$>G8@!
��$>G8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-
��$>G8@g+/��\5@1����/��?A�p��|#�?I}?q �@*	53333�K@2U
Iterator::Model::ParallelMapV2���Q��?!z|�h�:@)���Q��?1z|�h�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2�%䃎?!洛��:@)-C��6�?1�J3p��6@:Preprocessing2F
Iterator::Model�z6�>�?!1{}r�MD@)� �	�?1ϵ�
��+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate�(��0�?!�I�y� 6@)ŏ1w-!?1��<u 1+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceU���N@s?!Ҫe~�� @)U���N@s?1Ҫe~�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�St$���?!τ��%�M@)����Mbp?1O@�^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�QC#�@)�J�4a?1�QC#�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���QI�?!fMYS֔9@)����Mb`?1O@�^�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�Z�d<�X@Q��Ҵ���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	g+/��\5@g+/��\5@!g+/��\5@      ��!       "	����/��?����/��?!����/��?*      ��!       2	�p��|#�?�p��|#�?!�p��|#�?:	}?q �@}?q �@!}?q �@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�Z�d<�X@y��Ҵ���?�"5
sequential/dense/MatMulMatMulO"].ၣ?!O"].ၣ?0"C
%gradient_tape/sequential/dense/MatMulMatMulF��W�?!Jِql�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulF��W�?!m!���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulF��W�?!ȴ*���?"7
sequential/dense_1/MatMulMatMulF��W�?!��[Xy7�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulW��O���?!e�Z���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��I�D�?!���k���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradF��W�?!�D(����?")
sequential/CastCastF��W�?!����h��?"7
sequential/dense_2/MatMulMatMulF��W�?!O�,h$�?0Q      Y@Y>����/@aX�i��U@qvj��bX@yY�:�Q�?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 